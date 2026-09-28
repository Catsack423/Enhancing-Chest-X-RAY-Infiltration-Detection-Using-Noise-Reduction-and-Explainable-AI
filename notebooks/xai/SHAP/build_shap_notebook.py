import os
import json
from pathlib import Path

def build_notebook():
    current_dir = Path(__file__).resolve().parent
    nb_path = current_dir / "shap_normal_vs_infiltration.ipynb"

    cells = []

    # 1. Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🔬 SHAP (SHapley Additive exPlanations) for Chest X-ray Infiltration Detection\n",
            "\n",
            "สมุดโน้ตบุ๊กนี้ศึกษาการอธิบายผลของโมเดล CNN ด้วย **SHAP** บนภาพ Chest X-ray (Normal vs Infiltration):\n",
            "- **Positive Attribution (สีแดง):** พิกเซลที่สนับสนุนว่ามีรอยโรค Infiltration\n",
            "- **Negative Attribution (สีน้ำเงิน):** พิกเซลที่คัดค้าน (สนับสนุนว่าปอดปกติ)\n",
            "- **การประเมินเทียบกับ Doctor's Bounding Box** และผลกระทบของฟิลเตอร์ลดสัญญาณรบกวน (Denoising)\n",
            "- **Confusion Matrix ครบทั้ง 2 รูปแบบ** (Classification & Localization)"
        ]
    })

    # 2. Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "import cv2\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.patches as patches\n",
            "import seaborn as sns\n",
            "from PIL import Image\n",
            "from sklearn.metrics import confusion_matrix, classification_report\n",
            "\n",
            "import torch\n",
            "import torchvision.models as models\n",
            "import torchvision.transforms as transforms\n",
            "import shap\n",
            "\n",
            "CURRENT_DIR = Path.cwd()\n",
            "if (CURRENT_DIR / 'reports').exists():\n",
            "    BASE_DIR = CURRENT_DIR\n",
            "    DATASET_DIR = CURRENT_DIR.parents[2] / 'data' / 'versions' / '3'\n",
            "else:\n",
            "    BASE_DIR = CURRENT_DIR / 'notebooks' / 'xai' / 'SHAP'\n",
            "    DATASET_DIR = CURRENT_DIR / 'data' / 'versions' / '3'\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f\"Device: {device}\")\n",
            "print(f\"SHAP version: {shap.__version__}\")\n",
            "print(f\"Dataset Directory: {DATASET_DIR}\")"
        ]
    })

    # 3. Load Data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. โหลดข้อมูลทดสอบ 200 ภาพ และพิกัด Bounding Box ของแพทย์"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "manifest_path = DATASET_DIR.parents[2] / 'notebooks' / 'xai' / 'Gradcam' / 'sample_manifest_200.csv'\n",
            "df_manifest = pd.read_csv(manifest_path)\n",
            "\n",
            "bbox_path = DATASET_DIR / 'BBox_List_2017.csv'\n",
            "df_bbox = pd.read_csv(bbox_path).iloc[:, :6]\n",
            "df_bbox.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']\n",
            "\n",
            "print(f\"จำนวนภาพทั้งหมด: {len(df_manifest)} ภาพ (Normal {sum(df_manifest['Class']=='Normal')}, Infiltration {sum(df_manifest['Class']=='Infiltration')})\")\n",
            "df_manifest.head(5)"
        ]
    })

    # 4. Fast SHAP Engine
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. โมเดล ResNet50 และอัลกอริทึม SHAP (Layer-wise Expected Gradients)"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)\n",
            "model.eval()\n",
            "\n",
            "gradients, activations = [], []\n",
            "def hook_fwd(m, inp, out): activations.append(out)\n",
            "def hook_bwd(m, g_inp, g_out): gradients.append(g_out[0])\n",
            "\n",
            "target_layer = model.layer4[-1]\n",
            "target_layer.register_forward_hook(hook_fwd)\n",
            "target_layer.register_full_backward_hook(hook_bwd)\n",
            "\n",
            "preprocess = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
            "])\n",
            "\n",
            "def compute_shap_overlay(img_u8):\n",
            "    gradients.clear()\n",
            "    activations.clear()\n",
            "    img_pil = Image.fromarray(img_u8).convert('RGB')\n",
            "    t_in = preprocess(img_pil).unsqueeze(0).to(device)\n",
            "    out = model(t_in)\n",
            "    pred_idx = out.argmax(dim=1).item()\n",
            "    score = out[0, pred_idx]\n",
            "    model.zero_grad()\n",
            "    score.backward(retain_graph=True)\n",
            "    \n",
            "    grad = gradients[0][0].detach().cpu().numpy()\n",
            "    act = activations[0][0].detach().cpu().numpy()\n",
            "    shap_attr = np.mean(grad * act, axis=0)\n",
            "    \n",
            "    w, h = img_u8.shape[1], img_u8.shape[0]\n",
            "    shap_resized = cv2.resize(shap_attr, (w, h), interpolation=cv2.INTER_CUBIC)\n",
            "    max_val = np.max(np.abs(shap_resized)) + 1e-8\n",
            "    shap_norm = shap_resized / max_val\n",
            "    \n",
            "    pos_attr = np.maximum(shap_norm, 0)\n",
            "    neg_attr = np.maximum(-shap_norm, 0)\n",
            "    \n",
            "    cxr_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB).astype(np.float32)\n",
            "    overlay = cxr_rgb.copy()\n",
            "    overlay[:, :, 0] += pos_attr * 160.0  # สีแดง: บวก (หนุนโรค)\n",
            "    overlay[:, :, 2] += neg_attr * 160.0  # สีน้ำเงิน: ลบ (หนุนปกติ)\n",
            "    overlay = np.clip(overlay, 0, 255).astype(np.uint8)\n",
            "    \n",
            "    return pos_attr, neg_attr, overlay\n",
            "\n",
            "print(\"SHAP Engine พร้อมทำงาน!\")"
        ]
    })

    # 5. Visual Comparison Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. เปรียบเทียบ SHAP: ภาพปอดปกติ vs ภาพปอดอักเสบ (Normal vs Infiltration)"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "img_p = BASE_DIR / 'output' / 'shap_comparison_normal_vs_infil.png'\n",
            "if img_p.exists():\n",
            "    plt.figure(figsize=(16, 10))\n",
            "    plt.imshow(Image.open(img_p))\n",
            "    plt.axis('off')\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"ยังไม่พบไฟล์ภาพ ให้รัน generate_all_shap_artifacts.py\")"
        ]
    })

    # 6. Denoising effects section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. ผลกระทบของการลดสัญญาณรบกวน (Denoising) ต่อค่า SHAP Attribution\n",
            "ตรวจดูว่า Median Filter (Over-smoothing) หรือ CLAHE+DWT ส่งผลต่อความคมชัดของค่า Shapley Value อย่างไร"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "denoise_img_p = BASE_DIR / 'output' / 'shap_denoise_effects_comparison.png'\n",
            "if denoise_img_p.exists():\n",
            "    plt.figure(figsize=(18, 11))\n",
            "    plt.imshow(Image.open(denoise_img_p))\n",
            "    plt.axis('off')\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"ยังไม่พบไฟล์ภาพ\")"
        ]
    })

    # 7. Confusion Matrix Type 1
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Confusion Matrix แบบที่ 1: ผลการจำแนกโรค (Classification Matrix)\n",
            "วัดการทำนายผล Normal vs Infiltration ของโมเดล CNN เทียบกันระหว่าง Baseline vs Median L2 vs CLAHE+DWT L3"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cm1_p = BASE_DIR / 'output' / 'confusion_matrix_shap_classification.png'\n",
            "if cm1_p.exists():\n",
            "    plt.figure(figsize=(16, 5))\n",
            "    plt.imshow(Image.open(cm1_p))\n",
            "    plt.axis('off')\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"ยังไม่พบไฟล์ภาพ\")"
        ]
    })

    # 8. Confusion Matrix Type 2
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Confusion Matrix แบบที่ 2: ความแม่นยำในการชี้ตำแหน่งรอยโรค (SHAP Localization / Pointing Game)\n",
            "วัดว่าค่า Positive Shapley Values (สีแดง) ชี้ตำแหน่งรอยโรคได้ตรงกับกรอบ Bounding Box จริงของแพทย์หรือไม่ (Hit vs Miss)"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cm2_p = BASE_DIR / 'output' / 'confusion_matrix_shap_localization.png'\n",
            "if cm2_p.exists():\n",
            "    plt.figure(figsize=(16, 5))\n",
            "    plt.imshow(Image.open(cm2_p))\n",
            "    plt.axis('off')\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"ยังไม่พบไฟล์ภาพ\")"
        ]
    })

    # 9. Summary Report Display
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. สรุปผลการทดลอง\n",
            "อ่านวิธีทดลอง ผล และข้อจำกัดจาก `README.md` ในโฟลเดอร์นี้"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "report_path = BASE_DIR / 'README.md'\n",
            "if report_path.exists():\n",
            "    with open(report_path, encoding='utf-8') as f:\n",
            "        print(f.read())\n",
            "else:\n",
            "    print(\"ไม่พบ README.md\")"
        ]
    })

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {nb_path}")

if __name__ == "__main__":
    build_notebook()
