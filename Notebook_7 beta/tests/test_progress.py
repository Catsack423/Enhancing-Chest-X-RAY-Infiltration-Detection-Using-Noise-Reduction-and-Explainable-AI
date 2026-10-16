from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import importlib
import unittest
from unittest.mock import patch

from progress import LiveProgress, prepare_with_logs


class ProgressTests(unittest.TestCase):
    def test_heartbeat_and_exception_stop(self):
        output = StringIO()
        ready = Event()
        class Captured(LiveProgress):
            def heartbeat(self):
                super().heartbeat()
                ready.set()
        with redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with Captured("fixture", interval=.01) as progress:
                    with progress.stage("slow directory listing"):
                        self.assertTrue(ready.wait(1))
                        raise RuntimeError("fixture")
        self.assertFalse(progress.thread.is_alive())
        self.assertIn("slow directory listing", output.getvalue())
        self.assertIn("รอขั้นตอน", output.getvalue())
        self.assertIn("หยุด: RuntimeError", output.getvalue())

    def test_observer_preserves_results_sources_and_restores_functions(self):
        audit = importlib.import_module("v7.audit")
        colab = importlib.import_module("v7.colab")
        original_index, original_read, original_write = audit.image_index, audit.read_image, audit.write_json
        sources = {p: p.read_bytes() for p in Path(audit.__file__).parent.rglob("*.py")}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images_001/images").mkdir(parents=True)
            (root / "images_001/images/a.png").write_bytes(b"fixture")
            expected = original_index(root)
            def engine(*args, **kwargs):
                self.assertEqual(audit.image_index(root), expected)
                audit.write_json(root / "chunks/000000.json", {"records": [{"image": "a.png"}]})
                return "same result"
            output = StringIO()
            with patch.object(colab, "prepare_on_drive", side_effect=engine), redirect_stdout(output):
                self.assertEqual(prepare_with_logs(root, root, root), "same result")
            self.assertIn("พบ PNG 1", output.getvalue())
            self.assertIn("บันทึก checkpoint", output.getvalue())
            with patch.object(colab, "prepare_on_drive", side_effect=ValueError("stop")), redirect_stdout(StringIO()):
                with self.assertRaisesRegex(ValueError, "stop"):
                    prepare_with_logs(root, root, root)
        self.assertIs(audit.image_index, original_index)
        self.assertIs(audit.read_image, original_read)
        self.assertIs(audit.write_json, original_write)
        self.assertEqual(sources, {p: p.read_bytes() for p in sources})
