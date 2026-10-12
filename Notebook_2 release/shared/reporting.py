"""Reports are derived only from measured artifacts; pending rows contain no scores."""
from pathlib import Path
from os.path import relpath
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .denoising import read_gray
from .evaluation import classification_metrics
from .experiment import file_hash


def plot_confusion(matrix, path, title, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    matrix = np.asarray(matrix)
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix * 100.0, totals, out=np.zeros(matrix.shape, dtype=float), where=totals != 0)
    ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1], ["Normal", "Infiltration"])
    ax.set_yticks([0, 1], ["Normal", "Infiltration"])
    ax.set_xlabel("Predicted (count / % of actual class)")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, f"{value:,}\n({percentages[y, x]:.2f}%)", ha="center", va="center", color="black", bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})
    if own:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def save_overlay(raw, heatmap, box, path, signed=None):
    gray = np.repeat(raw[..., None], 3, axis=2).astype(np.float32)
    if signed is None:
        color = cv2.cvtColor(cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = gray * (1 - 0.45 * heatmap[..., None]) + color * (0.45 * heatmap[..., None])
    else:
        scale = float(np.max(np.abs(signed)))
        signed_display = signed / scale if scale else np.zeros_like(signed)
        overlay = gray.copy()
        overlay[..., 0] += np.maximum(signed_display, 0) * 140
        overlay[..., 2] += np.maximum(-signed_display, 0) * 140
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    x, y, w, h = box
    # Half-open raster bounds agree with bbox_mask; overlays preserve original dimensions.
    start = (max(0, int(np.floor(x))), max(0, int(np.floor(y))))
    end = (min(raw.shape[1]-1, int(np.ceil(x+w))-1), min(raw.shape[0]-1, int(np.ceil(y+h))-1))
    cv2.rectangle(overlay, start, end, (0, 255, 0), thickness=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Overlay PNG encoding failed")
    temporary = path.with_suffix(".tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def _markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict("records"):
        values = []
        for column in columns:
            value = row.get(column)
            values.append("—" if pd.isna(value) else f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def refresh_reports(exp):
    cnn_rows, matrices = [], {}
    statuses = {condition: {"condition_id": condition} for condition in exp.condition_ids}
    for condition in exp.condition_ids:
        folder = exp.results("CNN", condition)
        row = {"condition_id": condition, "status": "not_run", "n": None,
               "accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
        if (folder / "complete.json").exists():
            marker = json.loads((folder / "complete.json").read_text())
            exp.check_provenance(marker["provenance"], condition)
            if file_hash(folder / "best.pt") != marker["checkpoint_hash"]:
                raise ValueError("CNN checkpoint changed during reporting")
            stored = json.loads((folder / "metrics.json").read_text())
            predictions = pd.read_csv(folder / "test_predictions.csv")
            calculated = classification_metrics(predictions.label, predictions.prob_infiltration, exp.config["cnn"]["classification_threshold"])
            if calculated["confusion_matrix"] != stored["confusion_matrix"]:
                raise ValueError("Confusion matrix does not match measured predictions")
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                if not np.isclose(calculated[key], stored[key]):
                    raise ValueError(f"Stored CNN metric differs from predictions: {key}")
            row.update({k: v for k, v in stored.items() if k not in ("provenance", "confusion_matrix")})
            row["status"] = "complete"
            matrices[condition] = stored["confusion_matrix"]
            plot_confusion(stored["confusion_matrix"], folder / "confusion_matrix.png", condition)
        cnn_rows.append(row)
        statuses[condition]["CNN"] = row["status"]
    cnn = pd.DataFrame(cnn_rows)
    cnn_dir = exp.output_root / "CNN/comparison"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    cnn.to_csv(cnn_dir / "metrics_all_conditions.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, condition in zip(axes.flat, exp.condition_ids):
        if condition in matrices:
            plot_confusion(matrices[condition], None, condition, ax=ax)
        else:
            ax.set_title(condition)
            ax.text(0.5, 0.5, "NOT RUN", ha="center", va="center")
            ax.axis("off")
    mode = "SMOKE — pipeline verification only" if exp.config['run_mode'] == "smoke" else "MAIN"
    fig.suptitle(f"CNN — {mode} — {exp.config['data'].get('test_scope', 'official_full')} (configured n={exp.config['data']['expected_test_images']})")
    fig.tight_layout()
    fig.savefig(cnn_dir / "confusion_matrices_all.png", dpi=150)
    plt.close(fig)
    sections = ["# Notebook 2 — CNN / Grad-CAM / SHAP", "",
        "ผลในหน้านี้สร้างจากไฟล์ที่ประเมินจริง ช่องว่างหมายถึงยังไม่ได้รัน ไม่ใช่คะแนนศูนย์", "",
        f"คู่มือเตรียม Google Drive และเปิด Colab: [RUN_GUIDE.md]({Path(relpath(exp.root / 'RUN_GUIDE.md', exp.output_root)).as_posix()})", "",
        f"Run mode: **{exp.config['run_mode']}**; seed: **{exp.config['seed']}**. "
        "ผล smoke ใช้ตรวจการทำงานเท่านั้น ไม่ใช่ผลวิจัย", "",
        f"Train **{exp.config['data']['train_per_class']:,} ภาพ/คลาส**; "
        f"validation **{exp.config['data']['validation_per_class']:,} ภาพ/คลาส**. "
        f"ผลรอบหลักเก็บใต้ `{exp.config.get('output_subdirectory', '.')}/` ของโฟลเดอร์ Notebook_2", "",
        f"ฝึก epoch ละ **{exp.config['training']['images_per_epoch']} ภาพรวม**; "
        f"test **{exp.config['data']['expected_test_images']} ภาพ** ({exp.config['data'].get('test_scope', 'official_full')}). "
        "ผล test subset ไม่ใช่ผลประเมิน official test เต็ม", "",
        "อัปโหลด Notebook_2_colab.zip ไฟล์เดียวไป MyDrive/ แล้วเปิด notebook รุ่นใหม่ "
        "ชุดนี้มีเฉพาะภาพที่ใช้จริง; XAI เก็บต้นฉบับ ส่วน classification เก็บ working pixels 256×256 แบบ lossless", "",
        "## โครงสร้างและลำดับรัน", "",
        "1. CNN/cnn_comparison.ipynb: ตรวจข้อมูล ฝึก DAE และ CNN ทั้ง 10 เงื่อนไข บันทึก checkpoint", 
        "2. Grad-CAM/gradcam_comparison.ipynb: โหลด CNN ของแต่ละเงื่อนไขและประเมิน bbox",
        "3. SHAP/shap_comparison.ipynb: โหลด CNN เดียวกัน เก็บ signed attribution และประเมินเฉพาะค่าบวก", "",
        "shared/config.json เป็นค่าตั้งกลาง; shared/artifacts/manifests เป็นรายชื่อข้อมูลร่วม; "
        "แต่ละเกณฑ์เก็บผลรายเงื่อนไขใน results/ และผลเปรียบเทียบใน comparison/", "",
        "## CNN", "",
        _markdown_table(cnn, ["condition_id", "status", "n", "accuracy", "precision", "recall", "f1", "roc_auc"]), "",
        "![Confusion matrices — all conditions](CNN/comparison/confusion_matrices_all.png)", ""]
    for criterion in ("Grad-CAM", "SHAP"):
        summary_frames, detailed = [], []
        for condition in exp.condition_ids:
            folder = exp.results(criterion, condition)
            statuses[condition][criterion] = "not_run"
            if (folder / "complete.json").exists():
                marker = json.loads((folder / "complete.json").read_text())
                exp.check_provenance(marker["provenance"], condition)
                if file_hash(exp.results("CNN", condition) / "best.pt") != marker["cnn_checkpoint_hash"]:
                    raise ValueError("XAI results refer to a different CNN checkpoint")
                records = pd.read_csv(folder / "per_image_metrics.csv")
                expected = pd.read_csv(exp.artifacts / "manifests/xai.csv")
                if len(records) != len(expected) or records.image.duplicated().any() or set(records.image) != set(expected.image):
                    raise ValueError("Incomplete XAI records")
                summary_frames.append(pd.read_csv(folder / "summary.csv"))
                detailed.append(records)
                statuses[condition][criterion] = "complete"
        destination = exp.output_root / criterion / "comparison"
        destination.mkdir(parents=True, exist_ok=True)
        sections.extend([f"## {criterion}", ""])
        if summary_frames:
            summaries = pd.concat(summary_frames, ignore_index=True)
            records = pd.concat(detailed, ignore_index=True)
            summaries.to_csv(destination / "metrics_all_conditions.csv", index=False)
            records.to_csv(destination / "per_image_metrics_all_conditions.csv", index=False)
            summary_metrics = ("pointing_hit", "iou_0.3", "iou_0.5")
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            for ax, metric in zip(axes, summary_metrics):
                subset = summaries[(summaries.stratum == "overall") & (summaries.metric == metric)].set_index("condition_id")
                available = [c for c in exp.condition_ids if c in subset.index]
                rows = subset.loc[available]
                values = rows["mean"].to_numpy()
                ax.errorbar(np.arange(len(rows)), values, yerr=np.maximum(0, np.vstack([values - rows.ci_low, rows.ci_high - values])), fmt="o", capsize=4)
                ax.set_xticks(np.arange(len(rows)), available, rotation=70, ha="right")
                ax.set_title(metric + " — patient bootstrap 95% CI")
                ax.set_ylim(0, 1)
            fig.suptitle(f"{criterion} — {exp.config['run_mode'].upper()}" + (" — pipeline verification only" if exp.config['run_mode'] == "smoke" else ""))
            fig.tight_layout()
            fig.savefig(destination / "metrics_comparison.png", dpi=150)
            plt.close(fig)
            sections.extend([_markdown_table(summaries[summaries.stratum == "overall"], ["condition_id", "metric", "n_images", "n_patients", "mean", "ci_low", "ci_high"]), "", f"![{criterion} comparison]({criterion}/comparison/metrics_comparison.png)", ""])
            _visual_comparison(exp, criterion, destination)
        else:
            pd.DataFrame([{"condition_id": c, "status": "not_run"} for c in exp.condition_ids]).to_csv(destination / "metrics_all_conditions.csv", index=False)
            sections.extend(["ยังไม่มีผลที่ประเมินจริง", ""])
        local_readme = [f"# {criterion}", "", "ผลเปรียบเทียบทุกเงื่อนไขอยู่ใน comparison/metrics_all_conditions.csv", "",
                        "IoU ใช้ threshold 0.3 (หลัก) / 0.5 (เสริม); Pointing Game แบบ strict; bootstrap ระดับคนไข้ 2,000 รอบใน full mode", "",
                        "SHAP ประเมินเฉพาะ positive attribution ของ Infiltration logit; signed attribution เก็บใน heatmaps/*.npz และแสดงแดง/น้ำเงินใน overlays/" if criterion == "SHAP" else "Grad-CAM อธิบาย Infiltration logit ของ CNN ประจำเงื่อนไข แม้คำทำนายเป็น Normal", ""]
        (exp.output_root / criterion / "README.md").write_text("\n".join(local_readme), encoding="utf-8")
    status = pd.DataFrame(statuses.values())
    status.to_csv(exp.output_root / "completion_status.csv", index=False)
    sections.extend(["## ความครบถ้วน", "", _markdown_table(status, ["condition_id", "CNN", "Grad-CAM", "SHAP"]), "",
        "## วิธีตีความ", "",
        "Classification: Infiltration-only vs No Finding-only บน official test ที่กรองแล้วทั้งหมด "
        "XAI: bbox 123 ภาพ รวม mixed cases จึงเป็นคนละประชากรกับ classification test", "",
        "XAI แสดง Overall และแยก Pure / Mixed-high-risk / Mixed-low-risk ใน CSV "
        "CI สุ่มคนไข้ ไม่สุ่มภาพแยกกัน; heatmap ศูนย์นับ IoU=0 และ Pointing Game=Miss", "",
        "CLAHE+DWT ใช้ db1 และ DWT→CLAHE โดย clip/depth/threshold เปลี่ยนร่วมกัน จึงตีความเป็น joint presets "
        "DAE+CLAHE L1–L3 เปลี่ยน contrast หลัง DAE ไม่ใช่ความแรงของ DAE", "",
        "ข้อมูล checkpoint และ manifest hashes อยู่ในไฟล์ผลเพื่อป้องกันผลต่างการทดลองปะปนกัน "
        "label NIH มาจาก NLP และ bounding box เป็นกรอบคร่าว ๆ ไม่ใช่ lesion segmentation", ""])
    (exp.output_root / "README.md").write_text("\n".join(sections), encoding="utf-8")
    (exp.output_root / "CNN/README.md").write_text("# CNN\n\nฝึกและบันทึก checkpoint จริงครบ 10 เงื่อนไขก่อนรัน XAI\n\nตารางผล: comparison/metrics_all_conditions.csv\n\n![Confusion matrices](comparison/confusion_matrices_all.png)\n", encoding="utf-8")
    return status


def _visual_comparison(exp, criterion, destination):
    xai = pd.read_csv(exp.artifacts / "manifests/xai.csv")
    for row in xai.head(exp.config["xai"]["visual_examples"]).to_dict("records"):
        stem = row["image"].rsplit(".", 1)[0]
        fig, axes = plt.subplots(2, 5, figsize=(20, 9))
        for ax, condition in zip(axes.flat, exp.condition_ids):
            path = exp.results(criterion, condition) / "overlays" / f"{stem}.png"
            if path.is_file():
                rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
            else:
                ax.text(0.5, 0.5, "NOT RUN", ha="center", va="center")
            ax.set_title(condition)
            ax.axis("off")
        fig.suptitle(f"{criterion} — {row['image']} — {exp.config['run_mode'].upper()} — fixed example IDs")
        fig.tight_layout()
        fig.savefig(destination / f"heatmap_comparison_{stem}.png", dpi=150)
        plt.close(fig)
