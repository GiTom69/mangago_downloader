"""
Conversion functionality for the Mangago Downloader.
Supports converting downloaded manga images to PDF and CBZ formats.
"""
import io
import os
import zipfile
from typing import List, Optional
import img2pdf
from PIL import Image


def _normalize_image_for_pdf(image_path: str) -> bytes:
    """
    Prepare PDF-compatible image bytes in memory without modifying files on disk.
    Keeps plain RGB/grayscale JPEGs lossless without re-encoding, and normalizes
    WEBP, alpha-channel (RGBA/LA/PA), palette (P), or other modes to RGB PNG.
    """
    use_raw = False
    buf = io.BytesIO()
    with Image.open(image_path) as img:
        has_transparency = "transparency" in img.info
        if img.format == "JPEG" and img.mode in ("RGB", "L") and not has_transparency:
            use_raw = True
        else:
            if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and has_transparency):
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                rgb_img = background
            elif img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img
            rgb_img.save(buf, format="PNG")

    if use_raw:
        with open(image_path, "rb") as raw_f:
            return raw_f.read()
    return buf.getvalue()


def convert_to_pdf(
    chapter_dir: str,
    output_path: Optional[str] = None,
    delete_images: bool = False
) -> Optional[str]:
    """
    Convert chapter images to a high-quality PDF without re-encoding when possible,
    falling back to in-memory RGB normalization for WEBP/alpha/palette images
    rejected by img2pdf.
    """
    try:
        image_files = _get_image_files(chapter_dir)
        if not image_files:
            print(f"No images found in {chapter_dir}")
            return None
        
        if not output_path:
            chapter_name = os.path.basename(os.path.normpath(chapter_dir))
            output_path = os.path.join(chapter_dir, f"{chapter_name}.pdf")
        
        # Fast lossless path: pass file paths directly so img2pdf manages file handles
        # without re-encoding. Fall back to in-memory PIL normalization when img2pdf
        # rejects WEBP/alpha/palette images.
        try:
            pdf_bytes = img2pdf.convert(image_files)
        except Exception:
            normalized_images = [_normalize_image_for_pdf(path) for path in image_files]
            pdf_bytes = img2pdf.convert(normalized_images)

        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        
        if delete_images:
            for image_file in image_files:
                try:
                    os.remove(image_file)
                except OSError as e:
                    print(f"Warning: Failed to delete {image_file}: {e}")
        
        return output_path
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return None


def convert_to_cbz(
    chapter_dir: str,
    output_path: Optional[str] = None,
    delete_images: bool = False
) -> Optional[str]:
    """
    Convert chapter images to CBZ (Comic Book ZIP).
    """
    try:
        image_files = _get_image_files(chapter_dir)
        if not image_files:
            print(f"No images found in {chapter_dir}")
            return None
        
        if not output_path:
            chapter_name = os.path.basename(os.path.normpath(chapter_dir))
            output_path = os.path.join(chapter_dir, f"{chapter_name}.cbz")
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for image_file in image_files:
                arcname = os.path.basename(image_file)
                zipf.write(image_file, arcname)
        
        if delete_images:
            for image_file in image_files:
                try:
                    os.remove(image_file)
                except OSError as e:
                    print(f"Warning: Failed to delete {image_file}: {e}")
        
        return output_path
    except Exception as e:
        print(f"Error converting to CBZ: {e}")
        return None


def convert_manga_chapters(
    manga_dir: str,
    format: str = "pdf",
    delete_images: bool = False,
    chapter_dirs: Optional[List[str]] = None
) -> List[str]:
    """
    Convert chapters of a manga to the specified format.
    If chapter_dirs is provided, only convert those chapter directories.
    Otherwise, convert all chapter subdirectories under manga_dir.
    """
    created_files = []

    if chapter_dirs is None:
        try:
            target_dirs = [
                os.path.join(manga_dir, item)
                for item in os.listdir(manga_dir)
                if os.path.isdir(os.path.join(manga_dir, item))
            ]
        except FileNotFoundError:
            print(f"Error: Manga directory not found at {manga_dir}")
            return []
        target_dirs.sort()
    else:
        target_dirs = []
        seen = set()
        for chapter_dir in chapter_dirs:
            if not chapter_dir:
                continue
            resolved = chapter_dir
            if not os.path.isdir(resolved) and manga_dir:
                candidate = os.path.join(manga_dir, chapter_dir)
                if os.path.isdir(candidate):
                    resolved = candidate
            if not os.path.isdir(resolved):
                print(f"Warning: Chapter directory not found at {chapter_dir}")
                continue
            norm = os.path.normpath(os.path.abspath(resolved))
            if norm not in seen:
                seen.add(norm)
                target_dirs.append(resolved)

    for chapter_dir in target_dirs:
        try:
            if format.lower() == "pdf":
                output_file = convert_to_pdf(chapter_dir, delete_images=delete_images)
            elif format.lower() == "cbz":
                output_file = convert_to_cbz(chapter_dir, delete_images=delete_images)
            else:
                print(f"Unsupported format: {format}")
                continue
            
            if output_file:
                created_files.append(output_file)
                print(f"Converted {os.path.basename(os.path.normpath(chapter_dir))} to {os.path.basename(output_file)}")
        except Exception as e:
            print(f"Error converting {os.path.basename(os.path.normpath(chapter_dir))}: {e}")
    
    return created_files


def _image_sort_key(filepath: str):
    """Sort key that orders numeric stems numerically, with non-numeric stems after."""
    basename = os.path.basename(filepath)
    stem = os.path.splitext(basename)[0]
    try:
        return (0, int(stem), basename)
    except ValueError:
        return (1, 0, basename)


def _get_image_files(directory: str) -> List[str]:
    """
    Get all image files in a directory, sorted numerically.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    image_files = []
    
    try:
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(directory, filename))
    except FileNotFoundError:
        print(f"Directory not found: {directory}")

    # Sort numerically based on filename (e.g., 1.jpg, 2.jpg, 10.jpg)
    image_files.sort(key=_image_sort_key)
    
    return image_files