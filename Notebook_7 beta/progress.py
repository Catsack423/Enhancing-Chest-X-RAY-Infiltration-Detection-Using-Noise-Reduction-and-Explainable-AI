"""Colab console progress, kept outside training/audit source fingerprints."""
from contextlib import contextmanager, ExitStack
from functools import wraps
from pathlib import Path
from threading import Event, Lock, Thread
import importlib
import time


class LiveProgress:
    def __init__(self, label, interval=10):
        if interval <= 0:
            raise ValueError("Progress interval must be positive")
        self.label, self.interval = label, interval
        self.started = time.monotonic()
        self.phase = "เริ่มต้น / โหลดโมดูล"
        self.counts = {}
        self.lock, self.stop = Lock(), Event()

    def emit(self, message):
        line = f"[{self.label} +{time.monotonic()-self.started:.0f}s] {message}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="backslashreplace").decode("ascii"), flush=True)

    def update(self, **counts):
        with self.lock:
            self.counts.update(counts)

    def increment(self, name, count=1):
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + count

    def heartbeat(self):
        with self.lock:
            phase, counts = self.phase, dict(self.counts)
        suffix = "; ".join(f"{key}={value:,}" for key, value in counts.items())
        self.emit(f"รอขั้นตอน: {phase}" + (f"; {suffix}" if suffix else ""))

    def _watch(self):
        while not self.stop.wait(self.interval):
            self.heartbeat()

    def __enter__(self):
        self.emit("เริ่มงาน — แสดงสถานะทุก 10 วินาที; checkpoint อยู่บน Drive")
        self.thread = Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        self.emit("เสร็จแล้ว" if kind is None else f"หยุด: {kind.__name__}: {value}")

    @contextmanager
    def stage(self, label):
        with self.lock:
            previous, self.phase = self.phase, label
        started = time.monotonic()
        self.emit(label)
        try:
            yield
        finally:
            self.emit(f"จบขั้นตอน ({time.monotonic()-started:.1f}s): {label}")
            with self.lock:
                self.phase = previous


@contextmanager
def _replace(module, name, replacement):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield original
    finally:
        setattr(module, name, original)


def prepare_with_logs(root, data_root, work, workers=4, *, interval=10, zip_root=None):
    """Observe the installed v7 engine; preserve its signatures and cached results.

    Call after stopping the old preparation cell, never concurrently with that cell.
    Wrappers only log calls/results; they restore module functions even on failure.
    """
    with LiveProgress("v7 audit/selection", interval) as progress:
        colab = importlib.import_module("v7.colab")
        audit = importlib.import_module("v7.audit")
        prepare = importlib.import_module("v7.prepare")

        def staged(function, label):
            @wraps(function)
            def call(*args, **kwargs):
                with progress.stage(label):
                    return function(*args, **kwargs)
            return call

        original_index = audit.image_index
        @wraps(original_index)
        def index(*args, **kwargs):
            location = "Colab VM" if zip_root is not None else "DATA_ROOT"
            with progress.stage(f"ค้นหารายชื่อ PNG บน {location} (ยังไม่วิเคราะห์ noise)"):
                result = original_index(*args, **kwargs)
                progress.update(indexed_pngs=len(result))
                progress.emit(f"พบ PNG {len(result):,} ไฟล์")
                return result

        original_read = audit.read_image
        @wraps(original_read)
        def read(*args, **kwargs):
            try:
                result = original_read(*args, **kwargs)
            except Exception:
                progress.increment("read_errors_this_call")
                raise
            progress.increment("decoded_this_call")
            return result

        original_write = audit.write_json
        @wraps(original_write)
        def write(path, value, *args, **kwargs):
            result = original_write(path, value, *args, **kwargs)
            if Path(path).parent.name == "chunks" and "records" in value:
                count = len(value["records"])
                progress.increment("checkpoint_records_written_this_call", count)
                progress.emit(f"บันทึก checkpoint {Path(path).name}: {count:,} รายการ")
            return result

        with ExitStack() as stack:
            for module, name, replacement in (
                (colab, "selected_bundle", staged(colab.selected_bundle, "ตรวจชุดภาพที่บันทึกไว้ / SHA-256")),
                (colab, "ensure_source", staged(colab.ensure_source, "ตรวจ metadata และ official lists")),
                (audit, "audit", staged(audit.audit, "อ่าน metadata / checkpoint แล้ววิเคราะห์ noise")),
                (audit, "image_index", index),
                (audit, "read_image", read),
                (audit, "write_json", write),
                (prepare, "prepare", staged(prepare.prepare, "คัดภาพ / ตรวจ SHA-256 / คัดลอกภาพ / สร้าง package")),
                (prepare, "figures", staged(prepare.figures, "สร้างภาพตัวอย่างและกราฟสถิติ")),
            ):
                stack.enter_context(_replace(module, name, replacement))
            if zip_root is not None:
                with progress.stage("โหลด ZIP v3 / วิเคราะห์บน Colab / คัดภาพลง Drive"):
                    return colab.prepare_from_v3_zip(root, zip_root, data_root, work, workers=workers)
            return colab.prepare_on_drive(root, data_root, work, workers=workers)
