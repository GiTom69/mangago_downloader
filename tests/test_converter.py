"""Unit tests for src/converter.py."""
import os
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.converter import (
    _get_image_files,
    convert_manga_chapters,
    convert_to_cbz,
    convert_to_pdf,
)


def _create_test_image(path: Path, color: tuple = (255, 0, 0)) -> None:
    """Create a small valid RGB JPEG/PNG image on disk."""
    img = Image.new("RGB", (20, 20), color=color)
    img.save(path)


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _populate_chapter_dir(self, name: str, filenames: list[str]) -> Path:
        ch_dir = self.root / name
        ch_dir.mkdir(parents=True, exist_ok=True)
        for idx, fname in enumerate(filenames):
            _create_test_image(ch_dir / fname, color=(idx * 20 % 256, 50, 100))
        return ch_dir

    def test_numeric_image_sorting_in_get_image_files(self):
        ch_dir = self._populate_chapter_dir(
            "ch_sort", ["10.jpg", "2.jpg", "1.jpg", "11.jpg", "3.jpg", "cover.jpg"]
        )
        files = _get_image_files(str(ch_dir))
        basenames = [os.path.basename(f) for f in files]
        self.assertEqual(
            basenames,
            ["1.jpg", "2.jpg", "3.jpg", "10.jpg", "11.jpg", "cover.jpg"],
        )

    def test_convert_to_cbz_preserves_numeric_order(self):
        ch_dir = self._populate_chapter_dir("Chapter_1", ["10.jpg", "2.jpg", "1.jpg"])
        cbz_path = convert_to_cbz(str(ch_dir), delete_images=False)
        self.assertIsNotNone(cbz_path)
        self.assertTrue(os.path.exists(cbz_path))
        with zipfile.ZipFile(cbz_path, "r") as zf:
            self.assertEqual(zf.namelist(), ["1.jpg", "2.jpg", "10.jpg"])

    def test_convert_to_pdf_preserves_numeric_order_and_closes_handles(self):
        ch_dir = self._populate_chapter_dir("Chapter_2", ["10.jpg", "2.jpg", "1.jpg"])

        captured_args = []
        import img2pdf

        real_convert = img2pdf.convert

        def spy_convert(arg, *args, **kwargs):
            captured_args.append(list(arg))
            return real_convert(arg, *args, **kwargs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with patch("src.converter.img2pdf.convert", side_effect=spy_convert):
                pdf_path = convert_to_pdf(str(ch_dir), delete_images=True)

        self.assertIsNotNone(pdf_path)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertEqual(len(captured_args), 1)
        self.assertEqual(
            [os.path.basename(p) for p in captured_args[0]],
            ["1.jpg", "2.jpg", "10.jpg"],
        )
        # Ensure no ResourceWarning (leaked open file handles) was emitted
        resource_warnings = [
            w for w in caught if issubclass(w.category, ResourceWarning)
        ]
        self.assertEqual(resource_warnings, [])
        # Images should be deleted cleanly (would fail on Windows if handles leaked)
        self.assertFalse((ch_dir / "1.jpg").exists())
        self.assertFalse((ch_dir / "2.jpg").exists())
        self.assertFalse((ch_dir / "10.jpg").exists())

    def test_scoped_conversion_only_converts_and_deletes_selected_chapters(self):
        ch1 = self._populate_chapter_dir("Chapter_1", ["1.jpg", "2.jpg"])
        ch2 = self._populate_chapter_dir("Chapter_2", ["1.jpg", "2.jpg"])
        ch3 = self._populate_chapter_dir("Chapter_3", ["1.jpg", "2.jpg"])

        created = convert_manga_chapters(
            str(self.root),
            format="cbz",
            delete_images=True,
            chapter_dirs=[str(ch2)],
        )

        self.assertEqual(len(created), 1)
        self.assertTrue((ch2 / "Chapter_2.cbz").exists())
        # ch2 images deleted
        self.assertFalse((ch2 / "1.jpg").exists())
        self.assertFalse((ch2 / "2.jpg").exists())

        # ch1 and ch3 untouched (no cbz created, images intact)
        self.assertFalse((ch1 / "Chapter_1.cbz").exists())
        self.assertTrue((ch1 / "1.jpg").exists())
        self.assertTrue((ch1 / "2.jpg").exists())

        self.assertFalse((ch3 / "Chapter_3.cbz").exists())
        self.assertTrue((ch3 / "1.jpg").exists())
        self.assertTrue((ch3 / "2.jpg").exists())

    def test_convert_manga_chapters_default_converts_all(self):
        ch1 = self._populate_chapter_dir("Chapter_1", ["1.jpg"])
        ch2 = self._populate_chapter_dir("Chapter_2", ["1.jpg"])

        created = convert_manga_chapters(
            str(self.root), format="pdf", delete_images=False
        )
        self.assertEqual(len(created), 2)
        self.assertTrue((ch1 / "Chapter_1.pdf").exists())
        self.assertTrue((ch2 / "Chapter_2.pdf").exists())

    def test_convert_manga_chapters_empty_list_converts_nothing(self):
        ch1 = self._populate_chapter_dir("Chapter_1", ["1.jpg"])
        created = convert_manga_chapters(
            str(self.root), format="pdf", delete_images=True, chapter_dirs=[]
        )
        self.assertEqual(created, [])
        self.assertFalse((ch1 / "Chapter_1.pdf").exists())
        self.assertTrue((ch1 / "1.jpg").exists())

    def test_webp_and_rgba_png_convert_to_cbz_cleanly(self):
        ch_dir = self.root / "Chapter_Webp_CBZ"
        ch_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), (255, 10, 20)).save(ch_dir / "10.webp", format="WEBP")
        Image.new("RGBA", (24, 24), (10, 255, 20, 128)).save(ch_dir / "2.png", format="PNG")
        Image.new("RGB", (24, 24), (10, 20, 255)).save(ch_dir / "1.webp", format="WEBP")

        files = _get_image_files(str(ch_dir))
        self.assertEqual(
            [os.path.basename(f) for f in files],
            ["1.webp", "2.png", "10.webp"],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            cbz_path = convert_to_cbz(str(ch_dir), delete_images=True)

        self.assertIsNotNone(cbz_path)
        self.assertTrue(os.path.exists(cbz_path))
        with zipfile.ZipFile(cbz_path, "r") as zf:
            self.assertEqual(zf.namelist(), ["1.webp", "2.png", "10.webp"])
            self.assertIsNone(zf.testzip())

        resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(resource_warnings, [])
        self.assertFalse((ch_dir / "1.webp").exists())
        self.assertFalse((ch_dir / "2.png").exists())
        self.assertFalse((ch_dir / "10.webp").exists())

    def test_webp_rgba_la_and_palette_convert_to_pdf_with_zero_resource_warnings(self):
        ch_dir = self.root / "Chapter_Mixed_PDF"
        ch_dir.mkdir(parents=True, exist_ok=True)

        webp_path = ch_dir / "1.webp"
        rgba_path = ch_dir / "2.png"
        la_path = ch_dir / "3.png"
        pal_path = ch_dir / "4.png"
        jpg_path = ch_dir / "10.jpg"

        Image.new("RGB", (32, 32), (200, 30, 30)).save(webp_path, format="WEBP")
        Image.new("RGBA", (32, 32), (30, 200, 30, 120)).save(rgba_path, format="PNG")
        Image.new("LA", (32, 32), (180, 100)).save(la_path, format="PNG")
        pal_img = Image.new("P", (32, 32), 0)
        pal_img.putpalette([255, 0, 0, 0, 255, 0] + [0] * (254 * 3))
        pal_img.save(pal_path, format="PNG", transparency=0)
        Image.new("RGB", (32, 32), (30, 30, 200)).save(jpg_path, format="JPEG")

        original_webp_bytes = webp_path.read_bytes()
        original_rgba_bytes = rgba_path.read_bytes()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            pdf_path = convert_to_pdf(str(ch_dir), delete_images=False)

        self.assertIsNotNone(pdf_path)
        self.assertTrue(os.path.exists(pdf_path))
        pdf_bytes = Path(pdf_path).read_bytes()
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)

        # Files on disk must remain unmodified when delete_images=False
        self.assertEqual(webp_path.read_bytes(), original_webp_bytes)
        self.assertEqual(rgba_path.read_bytes(), original_rgba_bytes)

        resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(resource_warnings, [])

        # Also verify delete_images=True cleans up all mixed files without handle leaks
        with warnings.catch_warnings(record=True) as caught_del:
            warnings.simplefilter("always", ResourceWarning)
            pdf_path2 = convert_to_pdf(str(ch_dir), delete_images=True)

        self.assertIsNotNone(pdf_path2)
        self.assertEqual(
            [w for w in caught_del if issubclass(w.category, ResourceWarning)],
            [],
        )
        for p in (webp_path, rgba_path, la_path, pal_path, jpg_path):
            self.assertFalse(p.exists())

    def test_convert_to_pdf_uses_lossless_path_for_plain_jpegs(self):
        ch_dir = self._populate_chapter_dir("Chapter_Lossless", ["1.jpg", "2.jpg"])
        with patch("src.converter._normalize_image_for_pdf") as mock_norm:
            pdf_path = convert_to_pdf(str(ch_dir), delete_images=False)
        self.assertIsNotNone(pdf_path)
        self.assertTrue(Path(pdf_path).read_bytes().startswith(b"%PDF"))
        mock_norm.assert_not_called()

    def test_convert_to_pdf_pil_normalization_fallback_handles_webp_and_alpha(self):
        ch_dir = self.root / "Chapter_Fallback_PDF"
        ch_dir.mkdir(parents=True, exist_ok=True)
        webp_path = ch_dir / "1.webp"
        rgba_path = ch_dir / "2.png"
        jpg_path = ch_dir / "3.jpg"
        Image.new("RGB", (24, 24), (200, 10, 10)).save(webp_path, format="WEBP")
        Image.new("RGBA", (24, 24), (10, 200, 10, 128)).save(rgba_path, format="PNG")
        Image.new("RGB", (24, 24), (10, 10, 200)).save(jpg_path, format="JPEG")

        orig_webp = webp_path.read_bytes()
        orig_rgba = rgba_path.read_bytes()

        import img2pdf

        real_convert = img2pdf.convert
        calls = {"n": 0}

        def fail_first_convert(arg, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise img2pdf.AlphaChannelError("simulated alpha/webp rejection")
            return real_convert(arg, *args, **kwargs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with patch("src.converter.img2pdf.convert", side_effect=fail_first_convert):
                pdf_path = convert_to_pdf(str(ch_dir), delete_images=False)

        self.assertEqual(calls["n"], 2)
        self.assertIsNotNone(pdf_path)
        self.assertTrue(Path(pdf_path).read_bytes().startswith(b"%PDF"))
        # Files on disk must NOT be altered by fallback normalization
        self.assertEqual(webp_path.read_bytes(), orig_webp)
        self.assertEqual(rgba_path.read_bytes(), orig_rgba)
        self.assertEqual(
            [w for w in caught if issubclass(w.category, ResourceWarning)],
            [],
        )


if __name__ == "__main__":
    unittest.main()
