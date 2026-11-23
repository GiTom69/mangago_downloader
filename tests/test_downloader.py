"""Unit tests for src/downloader.py."""
import io
import os
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from PIL import Image
from selenium.common.exceptions import InvalidSessionIdException, TimeoutException

from src.downloader import (
    ChapterDownloader,
    UnderscoreHostnameAdapter,
    _build_chapter_dir_name,
    _build_referer,
    _detect_and_validate_image,
    _normalize_underscore_hostname,
    _replace_page_number_manhwa,
    fetch_chapter_image_urls,
    fetch_chapters_image_urls,
    get_chapter_list,
    init_driver,
)
from src.models import Chapter, Manga


def _make_image_bytes(fmt: str = "JPEG", mode: str = "RGB", color=(120, 40, 200)) -> bytes:
    buf = io.BytesIO()
    img = Image.new(mode, (16, 16), color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestPaginationURLs(unittest.TestCase):
    def test_replace_page_number_helper(self):
        url = "https://www.mangago.me/read-manga/sample/nml_chapter-396/pg-1/"
        self.assertEqual(
            _replace_page_number_manhwa(url, 6),
            "https://www.mangago.me/read-manga/sample/nml_chapter-396/pg-6/",
        )
        self.assertNotIn("/pg-1/6/", _replace_page_number_manhwa(url, 6))

    def test_fetch_chapter_image_urls_pg_pagination(self):
        start_url = "https://www.mangago.me/read-manga/sample/nml_chapter-396/pg-1/"
        visited_urls = []

        class FakeDriver:
            def __init__(self):
                self.current_url = ""

            def set_page_load_timeout(self, t):
                pass

            def get(self, url):
                visited_urls.append(url)
                self.current_url = url

            def quit(self):
                pass

        fake_driver = FakeDriver()

        # Simulate 6 pages (pg-1 through pg-6), then TimeoutException on pg-7
        page_counter = {"n": 0}

        class FakeWait:
            def __init__(self, driver, timeout):
                self.driver = driver

            def until(self, condition):
                page_counter["n"] += 1
                n = page_counter["n"]
                if n > 6:
                    raise TimeoutException("no more pages")
                img = MagicMock()
                img.get_attribute.return_value = f"https://cdn.example.com/page{n}.jpg"
                return img

        with (
            patch("src.downloader.webdriver.Chrome", return_value=fake_driver),
            patch("src.downloader.WebDriverWait", FakeWait),
        ):
            imgs = fetch_chapter_image_urls(start_url)

        expected_urls = [
            f"https://www.mangago.me/read-manga/sample/nml_chapter-396/pg-{i}/"
            for i in range(1, 8)
        ]
        self.assertEqual(visited_urls, expected_urls)
        self.assertFalse(any("/pg-1/" in u and u != start_url for u in visited_urls))
        self.assertEqual(
            imgs,
            [f"https://cdn.example.com/page{i}.jpg" for i in range(1, 7)],
        )

    def test_fetch_chapter_image_urls_fallback_pattern_preserved(self):
        start_url = "https://www.mangago.me/read-manga/sample/mi/12345/"
        visited_urls = []

        class FakeDriver:
            def __init__(self):
                self.current_url = ""

            def set_page_load_timeout(self, t):
                pass

            def get(self, url):
                visited_urls.append(url)
                self.current_url = url

            def quit(self):
                pass

        fake_driver = FakeDriver()
        wait_calls = {"n": 0}

        class FakeWait:
            def __init__(self, driver, timeout):
                pass

            def until(self, condition):
                wait_calls["n"] += 1
                if wait_calls["n"] == 1:
                    el = MagicMock()
                    el.text = "(1/3)"
                    return el
                img = MagicMock()
                img.get_attribute.return_value = (
                    f"https://cdn.example.com/p{wait_calls['n'] - 1}.jpg"
                )
                return img

        with (
            patch("src.downloader.webdriver.Chrome", return_value=fake_driver),
            patch("src.downloader.WebDriverWait", FakeWait),
        ):
            imgs = fetch_chapter_image_urls(start_url)

        self.assertEqual(
            visited_urls,
            [
                "https://www.mangago.me/read-manga/sample/mi/12345/",
                "https://www.mangago.me/read-manga/sample/mi/12345/2/",
                "https://www.mangago.me/read-manga/sample/mi/12345/3/",
            ],
        )
        self.assertEqual(len(imgs), 3)


class TestChapterListOrdering(unittest.TestCase):
    def test_get_chapter_list_preserves_site_order_with_specials_and_decimals(self):
        # Mangago chapter table is newest-first
        html = """
        <html><body>
        <table class="listing">
            <tr><td><a class="chico" href="/read-manga/test/c6/">Side Story 1</a></td></tr>
            <tr><td><a class="chico" href="/read-manga/test/c5/">Ch. 3</a></td></tr>
            <tr><td><a class="chico" href="/read-manga/test/c4/">Ch. 2.5</a></td></tr>
            <tr><td><a class="chico" href="/read-manga/test/c3/">Vol. 2 Special</a></td></tr>
            <tr><td><a class="chico" href="/read-manga/test/c2/">Ch. 2</a></td></tr>
            <tr><td><a class="chico" href="/read-manga/test/c1/">Special.</a></td></tr>
            <tr><td><a class="chico" href="/read-manga/test/c0/">Ch. 1</a></td></tr>
        </table>
        </body></html>
        """
        driver = MagicMock()
        driver.current_url = "https://www.mangago.me/read-manga/test/"
        driver.page_source = html

        with patch("src.downloader.WebDriverWait") as mock_wait:
            mock_wait.return_value.until.side_effect = TimeoutException("no button")
            chapters = get_chapter_list(driver)

        titles = [ch.title for ch in chapters]
        # Must be oldest-first (exact reverse of table), NOT re-sorted by parsed number
        self.assertEqual(
            titles,
            [
                "Ch. 1",
                "Special.",
                "Ch. 2",
                "Vol. 2 Special",
                "Ch. 2.5",
                "Ch. 3",
                "Side Story 1",
            ],
        )

    def test_duplicate_parsed_chapter_numbers_do_not_collide_on_disk(self):
        manga = Manga(title="Collision Test", url="https://www.mangago.me/read-manga/test/")
        ch_main = Chapter(
            number=1,
            title="Ch. 1",
            url="https://www.mangago.me/read-manga/test/ch-1/pg-1/",
            image_urls=["https://cdn.example.com/a.jpg"],
        )
        ch_side = Chapter(
            number=1,
            title="Side Story 1",
            url="https://www.mangago.me/read-manga/test/side-1/pg-1/",
            image_urls=["https://cdn.example.com/b.jpg"],
        )
        ch_same_title = Chapter(
            number=1,
            title="Ch. 1",
            url="https://www.mangago.me/read-manga/test/ch-1-alt/pg-1/",
            image_urls=["https://cdn.example.com/c.jpg"],
        )

        names = {
            _build_chapter_dir_name(ch_main),
            _build_chapter_dir_name(ch_side),
            _build_chapter_dir_name(ch_same_title),
        }
        self.assertEqual(len(names), 3)

        with tempfile.TemporaryDirectory() as tmp:
            downloader = ChapterDownloader(download_dir=tmp)
            try:
                def fake_download(url, path, referer):
                    Path(path).write_bytes(url.encode("utf-8"))
                    return True

                with patch.object(downloader, "_download_image", side_effect=fake_download):
                    r1 = downloader.download_chapter(manga, ch_main)
                    r2 = downloader.download_chapter(manga, ch_side)
                    r3 = downloader.download_chapter(manga, ch_same_title)

                self.assertTrue(r1.success and r2.success and r3.success)
                self.assertEqual(len({r1.file_path, r2.file_path, r3.file_path}), 3)
                # Verify each chapter's file was preserved and not overwritten
                self.assertEqual(
                    (Path(r1.file_path) / "001.jpg").read_bytes(),
                    b"https://cdn.example.com/a.jpg",
                )
                self.assertEqual(
                    (Path(r2.file_path) / "001.jpg").read_bytes(),
                    b"https://cdn.example.com/b.jpg",
                )
                self.assertEqual(
                    (Path(r3.file_path) / "001.jpg").read_bytes(),
                    b"https://cdn.example.com/c.jpg",
                )
            finally:
                downloader.close()


class TestChapterDownloadResult(unittest.TestCase):
    def test_download_chapter_returns_false_when_zero_images_download(self):
        manga = Manga(title="Fail Manga", url="https://www.mangago.me/read-manga/fail/")
        chapter = Chapter(
            number=1,
            title="Ch. 1",
            url="https://www.mangago.me/read-manga/fail/c1/",
            image_urls=[
                "https://cdn.example.com/1.jpg",
                "https://cdn.example.com/2.jpg",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            downloader = ChapterDownloader(download_dir=tmp)
            try:
                with patch.object(downloader, "_download_image", return_value=False):
                    result = downloader.download_chapter(manga, chapter)

                self.assertFalse(result.success)
                self.assertEqual(result.images_downloaded, 0)
                self.assertTrue(result.error_message)
            finally:
                downloader.close()

    def test_download_chapter_returns_false_when_no_image_urls(self):
        manga = Manga(title="Empty Manga", url="https://www.mangago.me/read-manga/empty/")
        chapter = Chapter(
            number=1,
            title="Ch. 1",
            url="https://www.mangago.me/read-manga/empty/c1/",
            image_urls=[],
        )

        with tempfile.TemporaryDirectory() as tmp:
            downloader = ChapterDownloader(download_dir=tmp)
            try:
                result = downloader.download_chapter(manga, chapter)
                self.assertFalse(result.success)
                self.assertEqual(result.images_downloaded, 0)
                self.assertTrue(result.error_message)
            finally:
                downloader.close()


class TestInitDriver(unittest.TestCase):
    def test_init_driver_has_no_invalid_non_headless_flag(self):
        captured_options = []

        def fake_chrome(options=None, **kwargs):
            captured_options.append(options)
            return MagicMock()

        with patch("src.downloader.webdriver.Chrome", side_effect=fake_chrome):
            init_driver()

        self.assertEqual(len(captured_options), 1)
        args = captured_options[0].arguments
        self.assertNotIn("--non-headless=new", args)


class TestUnderscoreHostnameAdapter(unittest.TestCase):
    def test_normalize_underscore_hostname(self):
        self.assertEqual(
            _normalize_underscore_hostname("iweb_11.mangapicgallery.com"),
            "iweb-11.mangapicgallery.com",
        )
        self.assertEqual(
            _normalize_underscore_hostname("iweb_6.mangapicgallery.com"),
            "iweb-6.mangapicgallery.com",
        )
        # Normal hostnames without underscores must NOT be altered
        self.assertIsNone(_normalize_underscore_hostname("iweb7.mangapicgallery.com"))
        self.assertIsNone(_normalize_underscore_hostname("www.mangago.me"))
        self.assertIsNone(_normalize_underscore_hostname("mangapicgallery.com"))
        # Multi-label or invalid hostnames must NOT be altered
        self.assertIsNone(_normalize_underscore_hostname("a.iweb_11.mangapicgallery.com"))
        self.assertIsNone(_normalize_underscore_hostname("a_1.b_2.example.com"))
        self.assertIsNone(_normalize_underscore_hostname("single_label"))
        self.assertIsNone(_normalize_underscore_hostname(""))
        self.assertIsNone(_normalize_underscore_hostname(None))

    def test_adapter_only_rewrites_underscore_subdomains_and_keeps_tls_verification(self):
        adapter = UnderscoreHostnameAdapter()
        try:
            normal_req = requests.Request(
                "GET", "https://iweb7.mangapicgallery.com/r/1.jpg"
            ).prepare()
            underscore_req = requests.Request(
                "GET", "https://iweb_11.mangapicgallery.com/r/1.jpg"
            ).prepare()

            sent_records = []

            def fake_super_send(self_adapter, request, *args, **kwargs):
                sent_records.append(
                    (
                        request.url,
                        self_adapter._assert_hostname,
                        kwargs.get("verify", True),
                    )
                )
                resp = requests.Response()
                resp.status_code = 200
                return resp

            with patch("requests.adapters.HTTPAdapter.send", autospec=True, side_effect=fake_super_send):
                adapter.send(normal_req)
                adapter.send(underscore_req)

            self.assertEqual(len(sent_records), 2)
            # Normal host passes untouched with no assert_hostname override
            self.assertEqual(sent_records[0], ("https://iweb7.mangapicgallery.com/r/1.jpg", None, True))
            # Underscore host uses hyphenated single-label assert_hostname and keeps verify=True
            self.assertEqual(
                sent_records[1],
                ("https://iweb_11.mangapicgallery.com/r/1.jpg", "iweb-11.mangapicgallery.com", True),
            )
            self.assertIn("iweb-11.mangapicgallery.com", adapter._sub_adapters)
            sub = adapter._sub_adapters["iweb-11.mangapicgallery.com"]
            self.assertEqual(
                sub.poolmanager.connection_pool_kw.get("assert_hostname"),
                "iweb-11.mangapicgallery.com",
            )
        finally:
            adapter.close()
        self.assertEqual(adapter._sub_adapters, {})

    def test_assert_hostname_does_not_blindly_accept_mismatched_certificates(self):
        from urllib3.util.ssl_match_hostname import CertificateError, match_hostname

        normalized = _normalize_underscore_hostname("iweb_11.mangapicgallery.com")
        self.assertEqual(normalized, "iweb-11.mangapicgallery.com")

        valid_wildcard_cert = {
            "subject": ((("commonName", "mangapicgallery.com"),),),
            "subjectAltName": (
                ("DNS", "mangapicgallery.com"),
                ("DNS", "*.mangapicgallery.com"),
            ),
        }
        wrong_domain_cert = {
            "subject": ((("commonName", "evil.example.com"),),),
            "subjectAltName": (
                ("DNS", "evil.example.com"),
                ("DNS", "*.evil.example.com"),
            ),
        }

        # Matches the legitimate wildcard SAN
        match_hostname(valid_wildcard_cert, normalized)
        # Rejects a certificate issued to a different domain
        with self.assertRaises(CertificateError):
            match_hostname(wrong_domain_cert, normalized)


class TestImageDownloadValidationAndRetries(unittest.TestCase):
    def test_build_referer_extracts_chapter_origin(self):
        self.assertEqual(
            _build_referer("https://www.mangago.me/read-manga/eleceed/nml_chapter-426/pg-1/"),
            "https://www.mangago.me/",
        )
        self.assertEqual(
            _build_referer("https://www.mangago.zone/chapter/123/456/"),
            "https://www.mangago.zone/",
        )
        self.assertEqual(_build_referer(""), "https://www.mangago.me/")

    def test_detect_and_validate_image_formats_and_rejects_html(self):
        self.assertEqual(_detect_and_validate_image(_make_image_bytes("WEBP")), ".webp")
        self.assertEqual(_detect_and_validate_image(_make_image_bytes("JPEG")), ".jpg")
        self.assertEqual(_detect_and_validate_image(_make_image_bytes("PNG")), ".png")
        self.assertIsNone(_detect_and_validate_image(b"<html><body>403 Forbidden</body></html>"))
        self.assertIsNone(_detect_and_validate_image(b""))

    def test_download_image_dynamic_referer_and_webp_saved_with_true_extension(self):
        webp_bytes = _make_image_bytes("WEBP")
        captured_headers = []

        def fake_get(url, headers=None, timeout=None):
            captured_headers.append(dict(headers or {}))
            resp = MagicMock()
            resp.status_code = 200
            resp.content = webp_bytes
            resp.headers = {"content-type": "image/jpeg"}
            resp.raise_for_status.return_value = None
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            downloader = ChapterDownloader(download_dir=tmp)
            try:
                sess = downloader._get_requests_session()
                target = Path(tmp) / "001.jpg"
                with patch.object(sess, "get", side_effect=fake_get):
                    ok = downloader._download_image(
                        "https://iweb_11.mangapicgallery.com/r/cspiclink/test.jpeg",
                        str(target),
                        "https://www.mangago.me/read-manga/eleceed/nml_chapter-426/pg-1/",
                    )

                self.assertTrue(ok)
                self.assertEqual(captured_headers[0]["Referer"], "https://www.mangago.me/")
                self.assertIn("image/webp", captured_headers[0]["Accept"])
                self.assertIn("Chrome/", captured_headers[0]["User-Agent"])
                # Saved as .webp, NOT .jpg
                self.assertTrue((Path(tmp) / "001.webp").is_file())
                self.assertFalse((Path(tmp) / "001.jpg").exists())
            finally:
                downloader.close()

    def test_download_image_rejects_html_and_retries_transient_failures(self):
        jpeg_bytes = _make_image_bytes("JPEG")

        with tempfile.TemporaryDirectory() as tmp:
            downloader = ChapterDownloader(download_dir=tmp)
            try:
                sess = downloader._get_requests_session()

                # 1. Invalid HTML bytes returned with HTTP 200 -> rejected after retries, no file written
                html_resp = MagicMock()
                html_resp.status_code = 200
                html_resp.content = b"<html>Access Denied</html>"
                html_resp.headers = {"content-type": "text/html"}
                html_resp.raise_for_status.return_value = None

                with (
                    patch.object(sess, "get", return_value=html_resp) as mock_get,
                    patch("src.downloader.time.sleep"),
                ):
                    ok = downloader._download_image(
                        "https://iweb_9.mangapicgallery.com/bad.jpg",
                        str(Path(tmp) / "001.jpg"),
                        "https://www.mangago.me/read-manga/eleceed/nml_chapter-418/pg-1/",
                        max_retries=3,
                    )

                self.assertFalse(ok)
                self.assertEqual(mock_get.call_count, 3)
                self.assertEqual(list(Path(tmp).iterdir()), [])

                # 2. Transient 503 then ConnectionError then valid JPEG -> succeeds on attempt 3
                resp_503 = MagicMock()
                resp_503.status_code = 503
                resp_ok = MagicMock()
                resp_ok.status_code = 200
                resp_ok.content = jpeg_bytes
                resp_ok.headers = {"content-type": "image/jpeg"}
                resp_ok.raise_for_status.return_value = None

                with (
                    patch.object(
                        sess,
                        "get",
                        side_effect=[resp_503, requests.ConnectionError("reset"), resp_ok],
                    ) as mock_get_retry,
                    patch("src.downloader.time.sleep"),
                ):
                    ok = downloader._download_image(
                        "https://iweb_9.mangapicgallery.com/ok.jpg",
                        str(Path(tmp) / "002.jpg"),
                        "https://www.mangago.me/read-manga/eleceed/nml_chapter-418/pg-1/",
                        max_retries=3,
                    )

                self.assertTrue(ok)
                self.assertEqual(mock_get_retry.call_count, 3)
                self.assertTrue((Path(tmp) / "002.jpg").is_file())

                # 3. Non-retryable 404 fails on first attempt without retrying
                resp_404 = MagicMock()
                resp_404.status_code = 404
                http_err = requests.HTTPError("404 Not Found", response=resp_404)
                resp_404.raise_for_status.side_effect = http_err

                with (
                    patch.object(sess, "get", return_value=resp_404) as mock_get_404,
                    patch("src.downloader.time.sleep"),
                ):
                    ok = downloader._download_image(
                        "https://iweb_9.mangapicgallery.com/missing.jpg",
                        str(Path(tmp) / "003.jpg"),
                        "https://www.mangago.me/read-manga/eleceed/nml_chapter-418/pg-1/",
                        max_retries=3,
                    )

                self.assertFalse(ok)
                self.assertEqual(mock_get_404.call_count, 1)
            finally:
                downloader.close()

    def test_download_chapter_resume_skips_existing_webp_and_reports_partial(self):
        manga = Manga(title="Eleceed", url="https://www.mangago.me/read-manga/eleceed/")
        chapter = Chapter(
            number=426,
            title="Ch. 426",
            url="https://www.mangago.me/read-manga/eleceed/nml_chapter-426/pg-1/",
            image_urls=["https://cdn.example.com/1.jpeg", "https://cdn.example.com/2.jpeg"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            downloader = ChapterDownloader(download_dir=tmp)
            try:
                ch_dir = Path(tmp) / "Eleceed" / _build_chapter_dir_name(chapter)
                ch_dir.mkdir(parents=True, exist_ok=True)
                (ch_dir / "001.webp").write_bytes(_make_image_bytes("WEBP"))

                with patch.object(downloader, "_download_image", return_value=False) as mock_dl:
                    res = downloader.download_chapter(manga, chapter)

                # Page 1 already exists as 001.webp, so only page 2 is attempted
                self.assertEqual(mock_dl.call_count, 1)
                self.assertTrue(res.success)
                self.assertEqual(res.images_downloaded, 1)
                self.assertIn("1/2", res.error_message or "")
            finally:
                downloader.close()


class TestDropdownPreloadAndRetries(unittest.TestCase):
    def test_dropdown_parsing_filters_other_chapters_and_skips_preloaded_pages(self):
        base = "https://www.mangago.me/read-manga/eleceed/nml_chapter-418"
        start_url = f"{base}/pg-1/"
        visited_urls = []

        dropdown_rows = [
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-1/", "text": "Page 1"},
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-2/", "text": "Page 2"},
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-3/", "text": "Page 3"},
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-4/", "text": "Page 4"},
            # Different chapter link must be filtered out
            {"href": "/read-manga/eleceed/nml_chapter-419/pg-1/", "text": "Page 1"},
            # Different domain must be filtered out
            {"href": "https://other.example.com/read-manga/eleceed/nml_chapter-418/pg-5/", "text": "Page 5"},
        ]

        # Visiting pg-1 preloads page1 + page2; visiting pg-3 preloads page3 + page4
        dom_by_url = {
            f"{base}/pg-1/": [
                {"id": "page1", "cls": "", "src": "https://iweb_9.mangapicgallery.com/p1.jpg"},
                {"id": "page2", "cls": "", "src": "https://iweb_9.mangapicgallery.com/p2.jpg"},
            ],
            f"{base}/pg-3/": [
                {"id": "page3", "cls": "", "src": "https://iweb_9.mangapicgallery.com/p3.jpg"},
                {"id": "page4", "cls": "", "src": "https://iweb_9.mangapicgallery.com/p4.jpg"},
            ],
        }

        class FakeDriver:
            def __init__(self):
                self.current_url = ""
                self.quit_called = False

            def set_page_load_timeout(self, t):
                pass

            def get(self, url):
                visited_urls.append(url)
                self.current_url = url

            def execute_script(self, script, *args):
                if "#dropdown-menu-page" in script:
                    return dropdown_rows
                if "img[id^='page']" in script:
                    return dom_by_url.get(self.current_url, [])
                return None

            def quit(self):
                self.quit_called = True

        driver = FakeDriver()

        class FakeWait:
            def __init__(self, drv, timeout):
                self.drv = drv

            def until(self, condition):
                return MagicMock()

        with patch("src.downloader.WebDriverWait", FakeWait):
            imgs = fetch_chapter_image_urls(start_url, driver=driver)

        # Must NOT quit an externally-provided driver
        self.assertFalse(driver.quit_called)
        # Preloaded pg-2 and pg-4 must be skipped!
        self.assertEqual(visited_urls, [f"{base}/pg-1/", f"{base}/pg-3/"])
        self.assertEqual(
            imgs,
            [
                "https://iweb_9.mangapicgallery.com/p1.jpg",
                "https://iweb_9.mangapicgallery.com/p2.jpg",
                "https://iweb_9.mangapicgallery.com/p3.jpg",
                "https://iweb_9.mangapicgallery.com/p4.jpg",
            ],
        )

    def test_per_page_retry_on_timeout_recovers_instead_of_breaking_early(self):
        base = "https://www.mangago.me/read-manga/eleceed/nml_chapter-418"
        start_url = f"{base}/pg-1/"
        stop_calls = []
        pg2_attempts = {"n": 0}

        dropdown_rows = [
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-1/", "text": "Page 1"},
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-2/", "text": "Page 2"},
            {"href": "/read-manga/eleceed/nml_chapter-418/pg-3/", "text": "Page 3"},
        ]

        class FakeDriver:
            def __init__(self):
                self.current_url = ""

            def get(self, url):
                self.current_url = url
                if url == f"{base}/pg-2/":
                    pg2_attempts["n"] += 1
                    if pg2_attempts["n"] == 1:
                        raise TimeoutException("renderer timeout on pg-2")

            def execute_script(self, script, *args):
                if "window.stop" in script:
                    stop_calls.append(self.current_url)
                    return None
                if "#dropdown-menu-page" in script:
                    return dropdown_rows
                if "img[id^='page']" in script:
                    if self.current_url == f"{base}/pg-1/":
                        return [{"id": "page1", "cls": "", "src": "https://cdn.example.com/1.jpg"}]
                    if self.current_url == f"{base}/pg-2/" and pg2_attempts["n"] >= 2:
                        return [{"id": "page2", "cls": "", "src": "https://cdn.example.com/2.jpg"}]
                    if self.current_url == f"{base}/pg-3/":
                        return [{"id": "page3", "cls": "", "src": "https://cdn.example.com/3.jpg"}]
                    return []
                return None

            def quit(self):
                pass

        driver = FakeDriver()

        class FakeWait:
            def __init__(self, drv, timeout):
                self.drv = drv

            def until(self, condition):
                if self.drv.current_url == f"{base}/pg-2/" and pg2_attempts["n"] == 1:
                    raise TimeoutException("img not ready on first attempt")
                return MagicMock()

        with (
            patch("src.downloader.WebDriverWait", FakeWait),
            patch("src.downloader.time.sleep"),
        ):
            imgs = fetch_chapter_image_urls(start_url, driver=driver)

        # Must retry pg-2, call window.stop(), and return ALL 3 pages (not stop at 1!)
        self.assertEqual(pg2_attempts["n"], 2)
        self.assertGreaterEqual(len(stop_calls), 1)
        self.assertEqual(
            imgs,
            [
                "https://cdn.example.com/1.jpg",
                "https://cdn.example.com/2.jpg",
                "https://cdn.example.com/3.jpg",
            ],
        )


class TestFetchChaptersDriverReuse(unittest.TestCase):
    def test_reuses_drivers_capped_at_max_workers_and_quits_in_finally(self):
        chapters = [
            Chapter(
                number=i,
                title=f"Ch. {i}",
                url=f"https://www.mangago.me/read-manga/eleceed/nml_chapter-{i}/pg-1/",
            )
            for i in range(1, 5)
        ]

        created_drivers = []
        used_driver_ids = []

        def fake_create_driver(url):
            drv = MagicMock(name=f"driver-{len(created_drivers) + 1}")
            created_drivers.append(drv)
            return drv

        def fake_fetch_single(url, driver=None):
            used_driver_ids.append(id(driver))
            return [f"{url}img1.jpg", f"{url}img2.jpg"]

        completed = []

        with (
            patch("src.downloader._create_driver_for_url", side_effect=fake_create_driver),
            patch("src.downloader.fetch_chapter_image_urls", side_effect=fake_fetch_single),
        ):
            fetch_chapters_image_urls(
                chapters,
                max_workers=2,
                on_chapter_complete=lambda ch, err: completed.append((ch.number, err)),
            )

        # Must create at most max_workers=2 drivers across 4 chapters
        self.assertLessEqual(len(created_drivers), 2)
        self.assertGreaterEqual(len(created_drivers), 1)
        # All 4 chapters received their image URLs and callback notifications
        self.assertEqual(sorted(n for n, err in completed), [1, 2, 3, 4])
        self.assertTrue(all(err is None for _, err in completed))
        for ch in chapters:
            self.assertEqual(len(ch.image_urls), 2)
        # Every created driver MUST be quit in finally
        for drv in created_drivers:
            drv.quit.assert_called_once()

    def test_recreates_driver_on_dead_session_and_quits_all_in_finally(self):
        chapters = [
            Chapter(
                number=1,
                title="Ch. 1",
                url="https://www.mangago.me/read-manga/eleceed/nml_chapter-1/pg-1/",
            ),
            Chapter(
                number=2,
                title="Ch. 2",
                url="https://www.mangago.me/read-manga/eleceed/nml_chapter-2/pg-1/",
            ),
        ]

        created_drivers = []

        def fake_create_driver(url):
            drv = MagicMock(name=f"driver-{len(created_drivers) + 1}")
            created_drivers.append(drv)
            return drv

        calls = {"n": 0}

        def fake_fetch_single(url, driver=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise InvalidSessionIdException("session deleted")
            return [f"{url}ok.jpg"]

        completed = []

        with (
            patch("src.downloader._create_driver_for_url", side_effect=fake_create_driver),
            patch("src.downloader.fetch_chapter_image_urls", side_effect=fake_fetch_single),
        ):
            fetch_chapters_image_urls(
                chapters,
                max_workers=1,
                on_chapter_complete=lambda ch, err: completed.append((ch.number, err)),
            )

        # First driver died, second driver was created and reused for both chapters
        self.assertEqual(len(created_drivers), 2)
        self.assertEqual(len(completed), 2)
        self.assertTrue(all(err is None for _, err in completed))
        self.assertEqual(chapters[0].image_urls, [f"{chapters[0].url}ok.jpg"])
        self.assertEqual(chapters[1].image_urls, [f"{chapters[1].url}ok.jpg"])
        for drv in created_drivers:
            drv.quit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
