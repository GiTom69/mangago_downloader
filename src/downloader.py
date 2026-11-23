"""
Downloader engine with threading support for the Mangago Downloader.
"""
import hashlib
import io
import logging
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from PIL import Image
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .models import Chapter, DownloadResult, Manga
from .utils import (
    DownloadError,
    NetworkError,
    ParsingError,
    SessionManager,
    create_directory,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

DEFAULT_PAGE_LOAD_TIMEOUT = 30
DEFAULT_IMAGE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
IMAGE_ACCEPT_HEADER = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"

PIL_FORMAT_TO_EXT: Dict[str, str] = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
    "PNG": ".png",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".webp", ".png", ".gif", ".bmp", ".tiff")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

AD_BLOCK_URL_PATTERNS = [
    "*googlesyndication*",
    "*doubleclick*",
    "*googletagmanager*",
    "*google-analytics*",
    "*pubfuture*",
    "*pubadx*",
    "*criteo*",
    "*crwdcntrl*",
    "*lotame*",
    "*popads*",
    "*adnxs*",
    "*rubiconproject*",
    "*amazon-adsystem*",
    "*facebook*",
    "*taboola*",
    "*outbrain*",
    "*mgid*",
    "*exoclick*",
    "*juicyads*",
    "*trafficjunky*",
    "*pubmatic*",
    "*openx*",
    "*casalemedia*",
    "*indexww*",
    "*id5-sync*",
    "*33across*",
    "*sharethrough*",
    "*quantserve*",
    "*badlandlispyippee*",
]


def _normalize_underscore_hostname(hostname: Optional[str]) -> Optional[str]:
    """
    If the leftmost DNS label contains an underscore (e.g. ``iweb_11.mangapicgallery.com``),
    return a single-label hyphenated equivalent (``iweb-11.mangapicgallery.com``) for
    wildcard SAN verification. Returns ``None`` for standard hostnames or multi-label
    underscore hostnames.
    """
    if not hostname or "." not in hostname:
        return None
    left, rest = hostname.split(".", 1)
    if not left or not rest or "_" not in left or "_" in rest:
        return None
    normalized_left = left.replace("_", "-")
    if not re.fullmatch(r"[A-Za-z0-9-]+", normalized_left):
        return None
    return f"{normalized_left}.{rest}"


class UnderscoreHostnameAdapter(HTTPAdapter):
    """
    HTTPAdapter that keeps full CA chain and wildcard SAN verification enabled while
    allowing single-label underscore CDN subdomains (e.g. ``iweb_11.mangapicgallery.com``)
    to match valid wildcard certificates (e.g. ``*.mangapicgallery.com``).

    Standard hostnames pass straight through untouched. Wrong-domain or untrusted
    certificates are still rejected with ``SSLError``.
    """

    def __init__(self, assert_hostname: Optional[str] = None, **kwargs):
        self._assert_hostname = assert_hostname
        self._sub_adapters: Dict[str, HTTPAdapter] = {}
        self._sub_lock = threading.Lock()
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        if self._assert_hostname:
            pool_kwargs["assert_hostname"] = self._assert_hostname
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        if self._assert_hostname:
            proxy_kwargs["assert_hostname"] = self._assert_hostname
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    def _get_sub_adapter(self, normalized_host: str) -> HTTPAdapter:
        with self._sub_lock:
            adapter = self._sub_adapters.get(normalized_host)
            if adapter is None:
                adapter = UnderscoreHostnameAdapter(assert_hostname=normalized_host)
                self._sub_adapters[normalized_host] = adapter
            return adapter

    def send(self, request, *args, **kwargs):
        if not self._assert_hostname and request.url:
            hostname = urlparse(request.url).hostname
            normalized = _normalize_underscore_hostname(hostname)
            if normalized:
                return self._get_sub_adapter(normalized).send(request, *args, **kwargs)
        return super().send(request, *args, **kwargs)

    def close(self):
        with self._sub_lock:
            for adapter in self._sub_adapters.values():
                adapter.close()
            self._sub_adapters.clear()
        super().close()


def _build_referer(chapter_referer: str) -> str:
    """Derive a clean origin Referer (``scheme://netloc/``) from the chapter URL."""
    parsed = urlparse(chapter_referer or "")
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/"
    return "https://www.mangago.me/"


def _detect_and_validate_image(content: bytes) -> Optional[str]:
    """
    Validate raw bytes with PIL and return the canonical file extension (e.g. '.webp', '.jpg').
    Returns ``None`` if bytes are not a valid image.
    """
    if not content:
        return None
    try:
        with Image.open(io.BytesIO(content)) as img:
            fmt = (img.format or "").upper()
            img.verify()
        return PIL_FORMAT_TO_EXT.get(fmt, ".jpg")
    except Exception:
        return None


def _find_existing_page_file(chapter_dir: str, page_index: int) -> Optional[str]:
    """Return path to an already-downloaded non-empty page file (``{i:03d}.*``), if any."""
    stem = f"{page_index:03d}"
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(chapter_dir, f"{stem}{ext}")
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _build_chapter_dir_name(chapter: Chapter) -> str:
    """Build a clean, deterministic, collision-resistant directory name for a chapter."""
    parts = [f"Chapter_{chapter.number}"]
    if chapter.title:
        safe_title = sanitize_filename(re.sub(r"\s+", "_", chapter.title.strip())).strip("._ ")
        if safe_title:
            parts.append(safe_title[:80])
    if chapter.url:
        url_hash = hashlib.sha1(chapter.url.encode("utf-8")).hexdigest()[:8]
        parts.append(url_hash)
    return sanitize_filename("_".join(parts)).rstrip(". ")


class ChapterDownloader:
    """
    Handles downloading of manga chapters with threading support.
    """

    def __init__(self, max_workers: int = 5, download_dir: str = "downloads"):
        self.max_workers = max_workers
        self.download_dir = download_dir
        self.session = SessionManager()
        self._thread_local = threading.local()
        self._requests_sessions: List[requests.Session] = []
        self._sessions_lock = threading.Lock()

    def _get_requests_session(self) -> requests.Session:
        """Return a thread-local requests.Session mounted with UnderscoreHostnameAdapter."""
        sess = getattr(self._thread_local, "requests_session", None)
        if sess is None:
            sess = requests.Session()
            adapter = UnderscoreHostnameAdapter()
            sess.mount("https://", adapter)
            sess.mount("http://", HTTPAdapter())
            self._thread_local.requests_session = sess
            with self._sessions_lock:
                self._requests_sessions.append(sess)
        return sess

    def download_chapter(self, manga: Manga, chapter: Chapter) -> DownloadResult:
        if not chapter.image_urls:
            return DownloadResult(chapter=chapter, success=False, error_message="No image URLs found.")

        manga_dir = os.path.join(self.download_dir, sanitize_filename(manga.title))
        create_directory(manga_dir)

        chapter_dir = os.path.join(manga_dir, _build_chapter_dir_name(chapter))
        create_directory(chapter_dir)

        total_images = len(chapter.image_urls)
        downloaded_count = 0
        for i, image_url in enumerate(chapter.image_urls, start=1):
            if _find_existing_page_file(chapter_dir, i):
                downloaded_count += 1
                continue
            image_filename = f"{i:03d}.jpg"
            image_path = os.path.join(chapter_dir, image_filename)
            if self._download_image(image_url, image_path, chapter.url):
                downloaded_count += 1

        if downloaded_count == 0:
            return DownloadResult(
                chapter=chapter,
                success=False,
                file_path=chapter_dir,
                error_message="Failed to download any images for this chapter.",
                images_downloaded=0,
            )

        error_message = None
        if downloaded_count < total_images:
            error_message = (
                f"Partial download: {downloaded_count}/{total_images} images downloaded."
            )
            logger.warning("Chapter %s: %s", chapter.number, error_message)

        return DownloadResult(
            chapter=chapter,
            success=True,
            file_path=chapter_dir,
            error_message=error_message,
            images_downloaded=downloaded_count,
        )

    def download_chapters(self, manga: Manga, chapters: List[Chapter]) -> List[DownloadResult]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chapter = {
                executor.submit(self.download_chapter, manga, chapter): chapter
                for chapter in chapters
            }
            for future in as_completed(future_to_chapter):
                results.append(future.result())
        return results

    def _download_image(
        self,
        image_url: str,
        image_path: str,
        chapter_referer: str,
        max_retries: int = 3,
    ) -> bool:
        target = Path(image_path)
        # Resume check: if any file with the same stem and known image ext already exists
        if target.parent.is_dir():
            for ext in IMAGE_EXTENSIONS:
                existing = target.with_suffix(ext)
                if existing.is_file() and existing.stat().st_size > 0:
                    return True

        referer = _build_referer(chapter_referer)
        headers = {
            "Referer": referer,
            "Accept": IMAGE_ACCEPT_HEADER,
            "User-Agent": DEFAULT_IMAGE_USER_AGENT,
        }
        sess = self._get_requests_session()
        last_error: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                response = sess.get(image_url, headers=headers, timeout=30)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < max_retries:
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                        continue
                    logger.warning(
                        "Failed to download %s after %d attempts: %s",
                        image_url,
                        max_retries,
                        last_error,
                    )
                    return False

                response.raise_for_status()
                content = response.content
                detected_ext = _detect_and_validate_image(content)
                if detected_ext is None:
                    content_type = response.headers.get("content-type", "unknown")
                    last_error = f"Invalid image content (Content-Type: {content_type}, {len(content)} bytes)"
                    if attempt < max_retries:
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                        continue
                    logger.warning("Rejected invalid image from %s: %s", image_url, last_error)
                    return False

                final_path = target.with_suffix(detected_ext)
                with open(final_path, "wb") as f:
                    f.write(content)
                return True

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                last_error = f"HTTP {status}: {e}"
                if status in RETRYABLE_STATUS_CODES and attempt < max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                logger.warning("HTTP error downloading %s: %s", image_url, last_error)
                return False
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                logger.warning(
                    "Failed to download %s after %d attempts: %s",
                    image_url,
                    max_retries,
                    last_error,
                )
                return False

        return False

    def close(self):
        self.session.close()
        with self._sessions_lock:
            for sess in self._requests_sessions:
                try:
                    sess.close()
                except Exception:
                    pass
            self._requests_sessions.clear()


def _replace_page_number_manhwa(url: str, page_number: int) -> str:
    """
    Replace the page number in a pg-N paginated URL.

    Args:
        url (str): The original chapter URL
        page_number (int): The new page number

    Returns:
        str: The URL with the updated page number
    """
    return re.sub(r"pg-\d+", f"pg-{page_number}", url)


def extract_chapter_id(url):
    match = re.search(r"chapter/\d+/(\d+)/", url)
    if match:
        return match.group(1)
    return None


def _configure_ad_blocking(driver: webdriver.Chrome) -> None:
    """Block known ad/tracker domains via Chrome DevTools Protocol when supported."""
    if not hasattr(driver, "execute_cdp_cmd"):
        return
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": AD_BLOCK_URL_PATTERNS})
    except Exception:
        # Ignore on mocked or non-CDP drivers
        pass


def _create_mangago_driver(page_load_timeout: int = DEFAULT_PAGE_LOAD_TIMEOUT) -> webdriver.Chrome:
    """Create a non-headless Chrome driver configured for mangago.me chapters."""
    options = webdriver.ChromeOptions()
    # Keep non-headless: Cloudflare blocks --headless=new on www.mangago.me.
    options.page_load_strategy = "eager"
    options.add_argument("--ignore-ssl-errors=true")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(page_load_timeout)
    _configure_ad_blocking(driver)
    return driver


def _is_driver_session_dead(exc: Exception) -> bool:
    """Return True if a Selenium exception indicates the browser session is no longer usable."""
    if isinstance(exc, InvalidSessionIdException):
        return True
    if isinstance(exc, WebDriverException):
        msg = str(exc).lower()
        dead_markers = (
            "invalid session id",
            "session deleted",
            "no such window",
            "chrome not reachable",
            "disconnected",
            "unable to connect to renderer",
        )
        return any(marker in msg for marker in dead_markers)
    return False


def _stop_page_load(driver: webdriver.Chrome) -> None:
    """Ask the browser to stop loading hanging subresources after a TimeoutException."""
    if hasattr(driver, "execute_script"):
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass


def _navigate_with_retry(
    driver: webdriver.Chrome,
    url: str,
    max_retries: int = 3,
    stop_on_timeout: bool = True,
) -> bool:
    """
    Navigate ``driver`` to ``url`` with retries on TimeoutException.
    Returns True if navigation finished cleanly, False if timed out on all attempts
    (caller may still check whether DOM elements loaded before failing).
    """
    for attempt in range(1, max_retries + 1):
        try:
            driver.get(url)
            return True
        except TimeoutException:
            if stop_on_timeout:
                _stop_page_load(driver)
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            return False
    return False


def _chapter_path_identity(chapter_url: str) -> Tuple[str, str]:
    """
    Return ``(base_path, mode)`` identifying a chapter path without its page suffix.
    """
    path = urlparse(chapter_url).path.rstrip("/")
    pg_match = re.match(r"^(.+)/pg-\d+$", path)
    if pg_match:
        return pg_match.group(1), "pg"
    mh_match = re.match(r"^(.+/c\d+)(?:/\d+)?$", path)
    if "/mh/" in path and mh_match:
        return mh_match.group(1), "mh"
    ch_page_match = re.match(r"^(.+/[^/]*chapter[^/]*)/\d+$", path, re.IGNORECASE)
    if ch_page_match:
        return ch_page_match.group(1), "plain"
    return path, "plain"


def _url_belongs_to_chapter(candidate_url: str, chapter_url: str) -> bool:
    """Return True if ``candidate_url`` is a page of the same chapter as ``chapter_url``."""
    cand_parsed = urlparse(candidate_url)
    ch_parsed = urlparse(chapter_url)
    if cand_parsed.netloc and ch_parsed.netloc and cand_parsed.netloc != ch_parsed.netloc:
        return False

    cand_path = cand_parsed.path.rstrip("/")
    base_path, mode = _chapter_path_identity(chapter_url)

    if mode == "pg":
        return bool(re.fullmatch(rf"{re.escape(base_path)}/pg-\d+", cand_path))
    if mode == "mh":
        return cand_path == base_path or bool(
            re.fullmatch(rf"{re.escape(base_path)}/\d+", cand_path)
        )
    return cand_path == base_path or bool(
        re.fullmatch(rf"{re.escape(base_path)}/(?:pg-)?\d+", cand_path)
    )


def _extract_page_num_from_url(url: str, fallback: int) -> int:
    path = urlparse(url).path.rstrip("/")
    pg_match = re.search(r"/pg-(\d+)$", path)
    if pg_match:
        return int(pg_match.group(1))
    return fallback


def _extract_dropdown_page_urls(
    driver: webdriver.Chrome,
    chapter_url: str,
) -> List[Tuple[int, str]]:
    """
    Parse ``#dropdown-menu-page a[href]`` on the current page and return ordered
    ``(page_num, absolute_url)`` pairs belonging to this chapter.
    """
    raw_links = None
    if hasattr(driver, "execute_script"):
        try:
            raw_links = driver.execute_script(
                """
                return Array.from(document.querySelectorAll('#dropdown-menu-page a[href]'))
                    .map(a => ({
                        href: a.getAttribute('href') || '',
                        text: (a.textContent || '').trim()
                    }));
                """
            )
        except Exception:
            raw_links = None

    if not isinstance(raw_links, list) or not raw_links:
        return []

    base_for_join = getattr(driver, "current_url", "") or chapter_url
    pages: List[Tuple[int, str]] = []
    seen_nums = set()
    seen_urls = set()

    for item in raw_links:
        if not isinstance(item, dict):
            continue
        href = item.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        abs_url = urljoin(base_for_join, href.strip())
        if not _url_belongs_to_chapter(abs_url, chapter_url):
            continue
        if abs_url in seen_urls:
            continue

        text = item.get("text") or ""
        text_match = re.search(r"page\s*(\d+)", text, re.IGNORECASE)
        if text_match:
            pnum = int(text_match.group(1))
        else:
            pnum = _extract_page_num_from_url(abs_url, len(pages) + 1)

        if pnum in seen_nums:
            continue
        seen_nums.add(pnum)
        seen_urls.add(abs_url)
        pages.append((pnum, abs_url))

    pages.sort(key=lambda pair: pair[0])
    return pages


def _extract_total_pages_from_js(driver: webdriver.Chrome) -> Optional[int]:
    """Read ``window.total_pages`` if exposed as a positive integer."""
    if not hasattr(driver, "execute_script"):
        return None
    try:
        val = driver.execute_script(
            "return (typeof window.total_pages === 'number' || typeof window.total_pages === 'string') ? parseInt(window.total_pages, 10) : null;"
        )
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    except Exception:
        pass
    return None


def _extract_img_url_from_element(img) -> Optional[str]:
    """Read best image URL attribute from a WebElement (src, data-src, lazy-src, currentSrc)."""
    for attr in ("src", "data-src", "lazy-src", "currentSrc"):
        try:
            val = img.get_attribute(attr)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _collect_page_images_from_dom(driver: webdriver.Chrome) -> Dict[int, str]:
    """
    Collect all ``img[id^='page']`` / ``img[class^='page']`` elements currently present in DOM,
    returning ``{page_number: image_url}``. Mangago preloads ``page{N+1}`` alongside ``page{N}``,
    allowing us to skip redundant page navigations.
    """
    if not hasattr(driver, "execute_script"):
        return {}
    try:
        rows = driver.execute_script(
            """
            return Array.from(document.querySelectorAll("img[id^='page'], img[class^='page']")).map(img => ({
                id: img.id || '',
                cls: img.className || '',
                src: img.getAttribute('src') || img.getAttribute('data-src') || img.getAttribute('lazy-src') || img.currentSrc || ''
            }));
            """
        )
    except Exception:
        return {}

    if not isinstance(rows, list):
        return {}

    collected: Dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = row.get("src")
        if not isinstance(src, str) or not src.strip():
            continue
        src = src.strip()
        if not src.startswith(("http://", "https://")):
            continue

        pid = row.get("id") or ""
        cls = row.get("cls") or ""
        match = re.search(r"\bpage(\d+)\b", pid) or re.search(r"\bpage(\d+)\b", cls)
        if match:
            pnum = int(match.group(1))
            if pnum not in collected:
                collected[pnum] = src
    return collected


def _build_page_urls_from_total(
    chapter_url: str,
    total_pages: int,
    is_pg_url: bool,
    is_manhwa_mh_url: bool,
) -> List[Tuple[int, str]]:
    """Build ordered ``(page_num, url)`` pairs when total_pages is known without dropdown links."""
    pages: List[Tuple[int, str]] = []
    if is_pg_url:
        for i in range(1, total_pages + 1):
            pages.append((i, _replace_page_number_manhwa(chapter_url, i)))
    elif is_manhwa_mh_url:
        match = re.search(r"(.*/c\d+/)", chapter_url)
        base_url = match.group(1) if match else chapter_url.rstrip("/") + "/"
        for i in range(1, total_pages + 1):
            if i == 1 and not re.search(r"/c\d+/\d+", base_url):
                pages.append((i, base_url))
            else:
                pages.append((i, f"{base_url}{i}/"))
    else:
        base = chapter_url.rstrip("/")
        for i in range(1, total_pages + 1):
            pages.append((i, chapter_url if i == 1 else f"{base}/{i}/"))
    return pages


def fetch_chapter_image_urls(
    chapter_url: str,
    driver: Optional[webdriver.Chrome] = None,
) -> List[str]:
    """
    Extract all image URLs from a paginated chapter.
    Supports various URL formats and uses Selenium for specific domains.

    If ``driver`` is supplied, it is reused and NOT closed when finished.
    If ``driver`` is ``None``, a driver is created and closed in ``finally``.
    """
    if not chapter_url:
        raise ParsingError("Chapter URL is invalid.")

    parsed_url = urlparse(chapter_url)
    domain = parsed_url.netloc

    img_urls: List[str] = []
    owns_driver = driver is None

    if domain in ["www.youhim.me", "www.mangago.zone"]:
        try:
            if owns_driver:
                driver = init_driver()
            current_url = chapter_url
            subpage_idx = 1

            while True:
                driver.get(current_url)

                # Ensure page ready
                wait_first_image(driver, timeout=25)

                # Scroll until all lazy images are in DOM
                imgs = load_all_images_on_subpage(
                    driver, initial_wait=10, pause=1.2, max_rounds=500, stable_rounds=3
                )

                # Add images from current subpage to list
                for img in sorted(imgs, key=sort_key_by_page_id):
                    src = _extract_img_url_from_element(img)
                    if src:
                        img_urls.append(src)

                # Navigate to next subpage (if exists)
                try:
                    next_el = driver.find_element(By.CSS_SELECTOR, "a.next_page")
                    href = next_el.get_attribute("href") or next_el.get_attribute("data-href")
                    if not href:
                        print("No next link visible. Finished chapter subpages.")
                        break

                    current_chapter_id = extract_chapter_id(driver.current_url)
                    next_url = urljoin(driver.current_url, href)
                    next_chapter_id = extract_chapter_id(next_url)

                    if current_chapter_id and next_chapter_id and current_chapter_id != next_chapter_id:
                        print(
                            f"Next page is a new chapter ({next_chapter_id}). "
                            "Stopping image collection for current chapter."
                        )
                        break

                    print(f"Navigating to next subpage: {next_url}")
                    current_url = next_url
                    subpage_idx += 1

                    # Apply 10-sec delay and scroll to bottom, then reset to top
                    print("Waiting 10 seconds and scrolling to bottom of new page...")
                    time.sleep(10)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    driver.execute_script("window.scrollTo(0, 0);")

                except NoSuchElementException:
                    print("No more next subpages. Finished chapter.")
                    break
                except TimeoutException:
                    print("Timeout while moving to next subpage; stopping image collection.")
                    break
                except Exception as e:
                    print(f"An error occurred during subpage navigation: {e}")
                    break

            return list(dict.fromkeys(img_urls))
        finally:
            if owns_driver and driver:
                close_driver(driver)
    else:
        try:
            if owns_driver:
                driver = _create_mangago_driver(page_load_timeout=DEFAULT_PAGE_LOAD_TIMEOUT)

            nav_ok = _navigate_with_retry(driver, chapter_url, max_retries=3)
            if not nav_ok and not _collect_page_images_from_dom(driver):
                raise TimeoutException(f"Timed out loading chapter URL: {chapter_url}")

            # Check if this is a vertical longstrip manhwa URL (pattern: /chapter/id1/id2/)
            is_vertical_longstrip = (
                "/chapter/" in chapter_url
                and len(chapter_url.split("/")) >= 6
                and not re.search(r"pg-\d+", chapter_url)
            )

            if is_vertical_longstrip:
                time.sleep(10)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                selectors = [
                    "img[id^='page']",
                    "img[class^='page']",
                    "img[src*='mangapicgallery.com']",
                ]

                found_images = set()
                for selector in selectors:
                    try:
                        img_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for img in img_elements:
                            img_url = _extract_img_url_from_element(img)
                            if img_url and img_url not in found_images:
                                alt_text = img.get_attribute("alt") or ""
                                if (
                                    img_url.endswith((".jpg", ".jpeg", ".png", ".webp"))
                                    and "arrow" not in alt_text.lower()
                                    and "icon" not in alt_text.lower()
                                    and "button" not in alt_text.lower()
                                ):
                                    img_urls.append(img_url)
                                    found_images.add(img_url)
                    except Exception:
                        continue
                return list(dict.fromkeys(img_urls))

            is_pg_url = bool(re.search(r"pg-\d+", chapter_url))
            is_manhwa_mh_url = "/mh/" in chapter_url and "/c" in chapter_url

            # Preferred path: discover exact page list from #dropdown-menu-page or window.total_pages
            pages = _extract_dropdown_page_urls(driver, chapter_url)
            if not pages:
                js_total = _extract_total_pages_from_js(driver)
                if js_total is not None:
                    pages = _build_page_urls_from_total(
                        chapter_url, js_total, is_pg_url, is_manhwa_mh_url
                    )

            if pages:
                collected: Dict[int, str] = {}
                # Wait for initial page image on the already-loaded first page
                first_page_num = pages[0][0]
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located(
                            (
                                By.CSS_SELECTOR,
                                f"img#page{first_page_num}, img.page{first_page_num}, img[id^='page']",
                            )
                        )
                    )
                except TimeoutException:
                    pass

                collected.update(_collect_page_images_from_dom(driver))

                for pnum, purl in pages:
                    if pnum in collected:
                        continue

                    for attempt in range(1, 4):
                        try:
                            if attempt > 1 or getattr(driver, "current_url", "") != purl:
                                try:
                                    driver.get(purl)
                                except TimeoutException:
                                    _stop_page_load(driver)

                            img_selector = f"img#page{pnum}, img.page{pnum}"
                            img = WebDriverWait(driver, 15).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, img_selector))
                            )
                            # Collect all visible/preloaded page images in DOM
                            dom_imgs = _collect_page_images_from_dom(driver)
                            for k, v in dom_imgs.items():
                                collected.setdefault(k, v)

                            if pnum not in collected:
                                img_url = _extract_img_url_from_element(img)
                                if img_url:
                                    collected[pnum] = img_url

                            if pnum in collected:
                                break
                        except TimeoutException:
                            _stop_page_load(driver)
                            if attempt < 3:
                                time.sleep(0.5 * attempt)
                                continue
                            logger.warning(
                                "Timed out fetching page %d (%s) after 3 attempts", pnum, purl
                            )
                        except Exception as e:
                            if _is_driver_session_dead(e):
                                raise
                            if attempt < 3:
                                time.sleep(0.5 * attempt)
                                continue
                            logger.warning(
                                "Error fetching page %d (%s): %s", pnum, purl, e
                            )

                if len(collected) < len(pages):
                    logger.warning(
                        "Collected %d/%d pages for %s", len(collected), len(pages), chapter_url
                    )

                return list(dict.fromkeys(collected[p] for p in sorted(collected)))

            # Fallback path when neither #dropdown-menu-page nor window.total_pages is present
            if is_pg_url or is_manhwa_mh_url:
                page_num = 1
                base_url = chapter_url

                if is_pg_url:
                    pg_match = re.search(r"pg-(\d+)", chapter_url)
                    if pg_match:
                        page_num = int(pg_match.group(1))
                elif is_manhwa_mh_url:
                    match = re.search(r"(.*/c\d+/)", chapter_url)
                    if match:
                        base_url = match.group(1)
                    pg_match = re.search(r"/c\d+/(\d+)", chapter_url)
                    if pg_match:
                        page_num = int(pg_match.group(1))

                collected_fallback: Dict[int, str] = _collect_page_images_from_dom(driver)

                while True:
                    if page_num in collected_fallback:
                        img_urls.append(collected_fallback[page_num])
                        page_num += 1
                        continue

                    if is_pg_url:
                        current_url = _replace_page_number_manhwa(chapter_url, page_num)
                    elif is_manhwa_mh_url:
                        if page_num == 1 and not re.search(r"/c\d+/\d+", base_url):
                            current_url = base_url
                        else:
                            current_url = f"{base_url}{page_num}/"
                    else:
                        break

                    if not current_url:
                        break

                    try:
                        if driver.current_url != current_url:
                            driver.get(current_url)

                        img_selector = f"img#page{page_num}, img.page{page_num}"
                        img = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, img_selector))
                        )
                        dom_imgs = _collect_page_images_from_dom(driver)
                        for k, v in dom_imgs.items():
                            collected_fallback.setdefault(k, v)

                        img_url = collected_fallback.get(page_num) or _extract_img_url_from_element(img)
                        if img_url:
                            collected_fallback[page_num] = img_url
                            img_urls.append(img_url)

                        page_num += 1

                    except TimeoutException:
                        break
                    except Exception as e:
                        if _is_driver_session_dead(e):
                            raise
                        break
            else:
                try:
                    page_info_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".multi_pg_tip.left"))
                    )
                    page_info = page_info_element.text
                    total_pages = int(page_info.split("/")[-1].replace(")", ""))
                except Exception:
                    total_pages = 1

                collected_legacy: Dict[int, str] = _collect_page_images_from_dom(driver)
                for i in range(1, total_pages + 1):
                    if i in collected_legacy:
                        img_urls.append(collected_legacy[i])
                        continue
                    url = chapter_url if i == 1 else f"{chapter_url.rstrip('/')}/{i}/"
                    for attempt in range(1, 4):
                        try:
                            if attempt > 1 or driver.current_url != url:
                                driver.get(url)

                            img = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, f"img#page{i}"))
                            )
                            dom_imgs = _collect_page_images_from_dom(driver)
                            for k, v in dom_imgs.items():
                                collected_legacy.setdefault(k, v)

                            img_url = collected_legacy.get(i) or _extract_img_url_from_element(img)
                            if img_url:
                                collected_legacy[i] = img_url
                                img_urls.append(img_url)
                            break
                        except TimeoutException:
                            _stop_page_load(driver)
                            if attempt < 3:
                                time.sleep(0.5 * attempt)
                                continue
                        except Exception as e:
                            if _is_driver_session_dead(e):
                                raise
                            break

            return list(dict.fromkeys(img_urls))
        finally:
            if owns_driver and driver:
                close_driver(driver)


def _create_driver_for_url(chapter_url: str) -> webdriver.Chrome:
    """Create the appropriate reusable Chrome driver for a chapter URL's domain."""
    domain = urlparse(chapter_url).netloc
    if domain in ["www.youhim.me", "www.mangago.zone"]:
        return init_driver()
    return _create_mangago_driver(page_load_timeout=DEFAULT_PAGE_LOAD_TIMEOUT)


def fetch_chapters_image_urls(
    chapters: List[Chapter],
    max_workers: int = 2,
    on_chapter_complete: Optional[Callable[[Chapter, Optional[Exception]], None]] = None,
) -> None:
    """
    Fetch image URLs for multiple chapters using a small pool of persistent Chrome drivers.

    Each worker thread launches at most one Chrome instance, reuses it across chapters
    (recreating it only if the session dies), assigns ``chapter.image_urls``, invokes
    ``on_chapter_complete(chapter, error)`` when provided, and quits its driver in ``finally``.
    """
    if not chapters:
        return

    worker_count = max(1, min(max_workers, len(chapters)))
    chapter_queue: "queue.Queue[Chapter]" = queue.Queue()
    for chapter in chapters:
        chapter_queue.put(chapter)

    def _worker() -> None:
        driver: Optional[webdriver.Chrome] = None
        driver_domain_group: Optional[str] = None
        try:
            while True:
                try:
                    chapter = chapter_queue.get_nowait()
                except queue.Empty:
                    break

                err: Optional[Exception] = None
                try:
                    if not chapter.url:
                        raise ParsingError("Chapter URL is invalid.")
                    domain = urlparse(chapter.url).netloc
                    needed_group = (
                        "zone"
                        if domain in ["www.youhim.me", "www.mangago.zone"]
                        else "default"
                    )
                    if driver is None or driver_domain_group != needed_group:
                        if driver is not None:
                            close_driver(driver)
                            driver = None
                        driver = _create_driver_for_url(chapter.url)
                        driver_domain_group = needed_group

                    try:
                        chapter.image_urls = fetch_chapter_image_urls(chapter.url, driver=driver)
                    except Exception as first_exc:
                        if _is_driver_session_dead(first_exc):
                            close_driver(driver)
                            driver = _create_driver_for_url(chapter.url)
                            driver_domain_group = needed_group
                            chapter.image_urls = fetch_chapter_image_urls(
                                chapter.url, driver=driver
                            )
                        else:
                            raise
                except Exception as exc:
                    err = exc
                finally:
                    chapter_queue.task_done()

                if on_chapter_complete is not None:
                    try:
                        on_chapter_complete(chapter, err)
                    except Exception as cb_exc:
                        logger.warning(
                            "on_chapter_complete callback failed for Chapter %s: %s",
                            chapter.number,
                            cb_exc,
                        )
        finally:
            if driver is not None:
                close_driver(driver)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_worker) for _ in range(worker_count)]
        for future in as_completed(futures):
            future.result()


def get_chapter_list(driver: webdriver.Chrome) -> List[Chapter]:
    """
    Get the list of chapters for a given manga from an existing driver instance.
    Automatically clicks 'show all chapters' button if present.
    Handles different domains.
    """
    parsed_url = urlparse(driver.current_url)
    domain = parsed_url.netloc

    if domain not in ["www.youhim.me", "www.mangago.zone"]:
        try:
            # Look for the "show all chapters" button
            show_all_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@onclick, 'showAllChapters()')]"))
            )
            # Click the button to show all chapters
            show_all_button.click()
            # Wait a bit for the page to update
            time.sleep(2)
        except Exception:
            # No "show all chapters" button found, proceed with existing chapters
            pass

    soup = BeautifulSoup(driver.page_source, "html.parser")
    chapter_table = soup.find("table", class_="listing")
    if not isinstance(chapter_table, Tag):
        raise DownloadError("Could not find chapter table.")

    chapters = []
    seen_urls = set()
    for row in chapter_table.find_all("tr"):
        link = row.find("a", class_="chico")
        if not link:
            continue

        title = link.get_text(strip=True)
        url = link.get("href")
        if not isinstance(url, str) or not url:
            continue
        if not url.startswith("http"):
            url = urljoin(driver.current_url, url)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        match = re.search(r"(?:Ch\.?|Chapter)\s*(\d+(?:\.\d+)?)", title, re.IGNORECASE)
        if not match:
            # Handles titles like "afterword.30.5"
            match = re.search(r"(\d+(?:\.\d+)?)", title)

        number = float(match.group(1)) if match else None
        if number is not None and number.is_integer():
            number = int(number)
        chapters.append(Chapter(number=number, title=title, url=url))

    # The table is newest-first; reverse it so index 1 is the oldest chapter.
    chapters.reverse()

    # Titles without a number (e.g. "Special.") get a unique number between their neighbours.
    previous_number = 0
    for index, chapter in enumerate(chapters):
        if chapter.number is None:
            next_number = next(
                (ch.number for ch in chapters[index + 1 :] if ch.number is not None),
                None,
            )
            if next_number is not None and next_number > previous_number:
                chapter.number = (previous_number + next_number) / 2
            else:
                chapter.number = previous_number + 0.5
        previous_number = chapter.number

    return chapters


def close_driver(driver: webdriver.Chrome):
    """Safely closes the WebDriver."""
    if driver:
        try:
            driver.quit()
        except Exception:
            pass


def init_driver():
    options = webdriver.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_argument("--ignore-ssl-errors=true")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    # Make viewport tall-ish so fewer scrolls are needed
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(DEFAULT_PAGE_LOAD_TIMEOUT)
    driver.set_window_size(1280, 2200)
    _configure_ad_blocking(driver)
    return driver


def wait_first_image(driver, timeout=20):
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "img[id^='page']"))
    )


def load_all_images_on_subpage(
    driver, initial_wait=10, pause=1.0, max_rounds=400, stable_rounds=3
):
    """
    Scrolls in steps until the number of images stops increasing for `stable_rounds`.
    This reliably triggers lazy-loading on long vertical strips.
    """
    # start at top
    driver.execute_script("window.scrollTo(0, 0);")
    # give scripts time to inject all <img> tags
    time.sleep(initial_wait)

    prev_count = 0
    stable = 0

    for _ in range(max_rounds):
        # Current images
        imgs = driver.find_elements(By.CSS_SELECTOR, "img[id^='page']")
        count = len(imgs)

        # If new images appeared, reset stability
        if count > prev_count:
            stable = 0
            prev_count = count
        else:
            stable += 1

        # Scroll: bring the last image fully into view; then advance by a viewport
        if imgs:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'end'});", imgs[-1])
            except Exception:
                pass

        driver.execute_script(
            """
            const step = Math.max(window.innerHeight, 900);
            window.scrollBy(0, step);
        """
        )

        time.sleep(pause)

        # If we've seen no growth for a while, try one hard jump to bottom once
        if stable == stable_rounds - 1:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause)

        if stable >= stable_rounds:
            break

    # Final collect
    imgs = driver.find_elements(By.CSS_SELECTOR, "img[id^='page']")
    return imgs


def sort_key_by_page_id(img_el):
    # Extract number from id="page123"
    try:
        pid = img_el.get_attribute("id") or ""
        match = re.search(r"page(\d+)", pid)
        if match:
            n = int(match.group(1))
            return n
        else:
            return 1_000_000  # fallback to push unknowns to end
    except Exception:
        return 1_000_000  # fallback to push unknowns to end
