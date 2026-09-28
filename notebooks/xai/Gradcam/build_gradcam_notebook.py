import os
import sys
import json
from pathlib import Path

def build_notebook():
    notebook_dir = Path(__file__).resolve().parent
    notebook_path = notebook_dir / "gradcam_normal_vs_infiltration.ipynb"
    
    cells = []
    
    # 1. Title & Intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🔬 การเปรียบเทียบภาพ Chest X-ray ปอดปกติ (Normal) vs ภาวะ Infiltration ด้วย Grad-CAM\n",
            "\n",
            "**วัตถุประสงค์ของสมุดโน้ตบุ๊กนี้:**\n",
            "1. ทำการศึกษาเปรียบเทียบเชิงอธิบายผล (Explainable AI - XAI) ระหว่างภาพปอดปกติ (**Normal / No Finding**) และภาพที่มีภาวะปอดอักเสบสารแทรกซึม (**Infiltration**)\n",
            "2. ใช้ชุดข้อมูลคลาสละ **100 ภาพ (รวม 200 ภาพ)** คัดเลือกจากคนไข้ที่ไม่ซ้ำกัน (Patient-level) โดยเคส Infiltration มีพิกัด Bounding Box จริงของแพทย์ประกอบครบถ้วน\n",
            "3. **ใช้ภาพเอกซเรย์ดิบ (Raw Chest X-ray)** โดยยังไม่ผ่านกระบวนการลดสัญญาณรบกวน (Noise Reduction) เพื่อใช้เป็นฐานอ้างอิง (Baseline Comparison)\n",
            "4. ใช้อัลกอริทึม **Grad-CAM (Gradient-weighted Class Activation Mapping)** บนโมเดล **ResNet50** เพื่อตรวจดูว่าโมเดลโฟกัสจุดใดในภาพปอดปกติ เทียบกับตำแหน่งรอยโรคจริงในภาพ Infiltration"
        ]
    })
    
    # 2. Imports & Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import glob\n",
            "import random\n",
            "from pathlib import Path\n",
            "\n",
            "import cv2\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.patches as patches\n",
            "from PIL import Image\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "import torchvision.models as models\n",
            "import torchvision.transforms as transforms\n",
            "\n",
            "# ตรวจสอบ Path\n",
            "CURRENT_DIR = Path.cwd()\n",
            "if (CURRENT_DIR / 'sample_manifest_200.csv').exists():\n",
            "    BASE_DIR = CURRENT_DIR\n",
            "    DATASET_DIR = CURRENT_DIR.parents[2] / 'data' / 'versions' / '3'\n",
            "else:\n",
            "    BASE_DIR = CURRENT_DIR / 'notebooks' / 'xai' / 'Gradcam'\n",
            "    DATASET_DIR = CURRENT_DIR / 'data' / 'versions' / '3'\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f\"Device: {device}\")\n",
            "print(f\"Dataset Directory: {DATASET_DIR}\")\n",
            "print(\"โหลดไลบรารี PyTorch, Torchvision, OpenCV สำเร็จ!\")"
        ]
    })

    # 3. Load Manifest & BBox
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. โหลดข้อมูลชุดทดลอง 200 ภาพ (Normal 100 ภาพ, Infiltration 100 ภาพ)"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# โหลด Manifest รายการภาพ 200 ภาพ\n",
            "manifest_path = BASE_DIR / 'sample_manifest_200.csv'\n",
            "df_manifest = pd.read_csv(manifest_path)\n",
            "\n",
            "# โหลด BBox จริงของแพทย์จาก BBox_List_2017.csv\n",
            "bbox_path = DATASET_DIR / 'BBox_List_2017.csv'\n",
            "df_bbox = pd.read_csv(bbox_path).iloc[:, :6]\n",
            "df_bbox.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']\n",
            "\n",
            "print(f\"จำนวนภาพในชุดทดลอง: {len(df_manifest)} ภาพ\")\n",
            "print(df_manifest['Class'].value_counts())\n",
            "print(f\"\\nจำนวนภาพ Infiltration ที่มี BBox อ้างอิง: {df_manifest[df_manifest['Class'] == 'Infiltration']['Has_BBox'].sum()} ภาพ\")\n",
            "df_manifest.head(6)"
        ]
    })

    # 4. ResNet50 & Grad-CAM Class
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. นิยามคลาส Grad-CAM สำหรับสถาปัตยกรรม ResNet50\n",
            "\n",
            "Grad-CAM คำนวณน้ำหนักความสำคัญของ Feature Map ในชั้น Convolution สุดท้าย (`layer4[-1]` ของ ResNet50) โดยใช้ Gradient ของคะแนนคลาสเป้าหมายเทียบกับ Activation Map:\n",
            "\n",
            "$$\\alpha_k^c = \\frac{1}{Z} \\sum_{i}\\sum_{j} \\frac{\\partial Y^c}{\\partial A_{i,j}^k}$$\n",
            "\n",
            "$$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left( \\sum_k \\alpha_k^c A^k \\right)$$"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class ResNet50GradCAM:\n",
            "    def __init__(self, model, target_layer=None):\n",
            "        self.model = model.to(device)\n",
            "        self.model.eval()\n",
            "        self.target_layer = target_layer if target_layer is not None else self.model.layer4[-1]\n",
            "        \n",
            "        self.gradients = []\n",
            "        self.activations = []\n",
            "        \n",
            "        # ติดตั้ง Hooks เพื่อดึง Activation และ Gradient\n",
            "        self.target_layer.register_forward_hook(self._save_activation)\n",
            "        self.target_layer.register_full_backward_hook(self._save_gradient)\n",
            "        \n",
            "    def _save_activation(self, module, inp, out):\n",
            "        self.activations.append(out)\n",
            "        \n",
            "    def _save_gradient(self, module, grad_inp, grad_out):\n",
            "        self.gradients.append(grad_out[0])\n",
            "        \n",
            "    def generate_heatmap(self, input_tensor, class_idx=None):\n",
            "        self.gradients.clear()\n",
            "        self.activations.clear()\n",
            "        \n",
            "        # Forward pass\n",
            "        output = self.model(input_tensor)\n",
            "        if class_idx is None:\n",
            "            class_idx = output.argmax(dim=1).item()\n",
            "            \n",
            "        # Backward pass หา gradient เทียบกับคลาสเป้าหมาย\n",
            "        self.model.zero_grad()\n",
            "        score = output[:, class_idx]\n",
            "        score.backward(retain_graph=True)\n",
            "        \n",
            "        # คำนวณ GAP ของ Gradients\n",
            "        grad = self.gradients[0][0].detach()\n",
            "        act = self.activations[0][0].detach()\n",
            "        weights = torch.mean(grad, dim=(1, 2), keepdim=True)\n",
            "        \n",
            "        # รวม Linear Combination แล้วผ่าน ReLU\n",
            "        cam = torch.sum(weights * act, dim=0).clamp(min=0).cpu().numpy()\n",
            "        \n",
            "        # Normalize สู่ช่วง 0 ถึง 1\n",
            "        if cam.max() > cam.min():\n",
            "            cam = (cam - cam.min()) / (cam.max() - cam.min())\n",
            "        else:\n",
            "            cam = np.zeros_like(cam)\n",
            "            \n",
            "        return cam, class_idx\n",
            "\n",
            "# โหลดโมเดล ResNet50 Pretrained ImageNet\n",
            "model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)\n",
            "gradcam = ResNet50GradCAM(model)\n",
            "\n",
            "# Pipeline เตรียมภาพเข้าสู่ ResNet50\n",
            "transform = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
            "])\n",
            "\n",
            "print(\"ตั้งค่าโมเดล ResNet50 และคลาส Grad-CAM พร้อมใช้งานแล้ว!\")"
        ]
    })

    # 5. Pipeline visualization function
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. ฟังก์ชันสร้างภาพ Overlay (Grad-CAM + Chest X-ray)"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def overlay_gradcam_on_cxr(img_pil, cam, alpha=0.45, colormap=cv2.COLORMAP_JET):\n",
            "    \"\"\"\n",
            "    นำ Heatmap ของ Grad-CAM มาขยายเท่าขนาดภาพต้นฉบับ แล้วซ้อนทับ (Blend) บนภาพ Chest X-ray\n",
            "    \"\"\"\n",
            "    w, h = img_pil.size\n",
            "    cam_resized = cv2.resize(cam, (w, h))\n",
            "    heatmap = np.uint8(255 * cam_resized)\n",
            "    heatmap_colored = cv2.applyColorMap(heatmap, colormap)\n",
            "    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)\n",
            "    \n",
            "    cxr_rgb = np.array(img_pil.convert('RGB'))\n",
            "    blended = np.uint8(alpha * heatmap_colored + (1 - alpha) * cxr_rgb)\n",
            "    return blended, cam_resized\n",
            "\n",
            "def process_image(img_path):\n",
            "    img_pil = Image.open(img_path)\n",
            "    input_tensor = transform(img_pil.convert('RGB')).unsqueeze(0).to(device)\n",
            "    cam, class_idx = gradcam.generate_heatmap(input_tensor)\n",
            "    overlay, cam_full = overlay_gradcam_on_cxr(img_pil, cam)\n",
            "    return img_pil, cam_full, overlay"
        ]
    })

    # 6. Side-by-Side Comparison: Normal vs Infiltration
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. เปรียบเทียบแบบเคสต่อเคส (Pairwise Comparison: Normal vs Infiltration)\n",
            "แสดงภาพต้นฉบับ, Heatmap, ภาพซ้อนทับ (Grad-CAM + CXR) และกรอบ Bounding Box จริงของแพทย์ (สำหรับเคส Infiltration)"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def compare_normal_vs_infil(normal_idx=0, infil_idx=0):\n",
            "    normal_row = df_manifest[df_manifest['Class'] == 'Normal'].iloc[normal_idx]\n",
            "    infil_row = df_manifest[df_manifest['Class'] == 'Infiltration'].iloc[infil_idx]\n",
            "    \n",
            "    norm_path = DATASET_DIR / normal_row['Relative_Path']\n",
            "    infil_path = DATASET_DIR / infil_row['Relative_Path']\n",
            "    \n",
            "    # ประมวลผลทั้งสองภาพ\n",
            "    norm_img, norm_cam, norm_overlay = process_image(norm_path)\n",
            "    infil_img, infil_cam, infil_overlay = process_image(infil_path)\n",
            "    \n",
            "    # ดึง Bounding Box ของเคส Infiltration\n",
            "    infil_name = infil_row['Image Index']\n",
            "    boxes = df_bbox[df_bbox['Image Index'] == infil_name]\n",
            "    \n",
            "    # พล็อตเปรียบเทียบ 2 แถว 3 คอลัมน์\n",
            "    fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
            "    \n",
            "    # แถวบน: ภาพปอดปกติ (Normal / No Finding)\n",
            "    axes[0, 0].imshow(norm_img, cmap='gray')\n",
            "    axes[0, 0].set_title(f\"[NORMAL] Raw CXR\\n{normal_row['Image Index']}\", fontsize=11, weight='bold')\n",
            "    axes[0, 0].axis('off')\n",
            "    \n",
            "    axes[0, 1].imshow(norm_cam, cmap='jet')\n",
            "    axes[0, 1].set_title(\"[NORMAL] Grad-CAM Heatmap\", fontsize=11, weight='bold')\n",
            "    axes[0, 1].axis('off')\n",
            "    \n",
            "    axes[0, 2].imshow(norm_overlay)\n",
            "    axes[0, 2].set_title(\"[NORMAL] Overlay (CXR + Grad-CAM)\", fontsize=11, weight='bold')\n",
            "    axes[0, 2].axis('off')\n",
            "    \n",
            "    # แถวล่าง: ภาพปอดอักเสบ (Infiltration)\n",
            "    axes[1, 0].imshow(infil_img, cmap='gray')\n",
            "    axes[1, 0].set_title(f\"[INFILTRATION] Raw CXR\\n{infil_name}\", fontsize=11, weight='bold')\n",
            "    axes[1, 0].axis('off')\n",
            "    \n",
            "    axes[1, 1].imshow(infil_cam, cmap='jet')\n",
            "    axes[1, 1].set_title(\"[INFILTRATION] Grad-CAM Heatmap\", fontsize=11, weight='bold')\n",
            "    axes[1, 1].axis('off')\n",
            "    \n",
            "    # แถวล่างขวา: Overlay พร้อมตีกรอบรอยโรคจริง (Doctor's Ground-Truth BBox)\n",
            "    axes[1, 2].imshow(infil_overlay)\n",
            "    for _, b in boxes.iterrows():\n",
            "        bx, by, bw, bh = float(b['x']), float(b['y']), float(b['w']), float(b['h'])\n",
            "        rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor='#00FF66', facecolor='none', linestyle='--')\n",
            "        axes[1, 2].add_patch(rect)\n",
            "        axes[1, 2].text(\n",
            "            bx, max(0, by - 10), \"BBox: Infiltrate\",\n",
            "            color='black', fontsize=9, weight='bold',\n",
            "            bbox=dict(boxstyle='square,pad=0.2', facecolor='#00FF66', edgecolor='none', alpha=0.9)\n",
            "        )\n",
            "    axes[1, 2].set_title(\"[INFILTRATION] Overlay + Doctor's BBox (Green)\", fontsize=11, weight='bold')\n",
            "    axes[1, 2].axis('off')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "# รันตัวอย่างคู่แรก\n",
            "compare_normal_vs_infil(normal_idx=0, infil_idx=0)"
        ]
    })

    # 7. Multi-sample grid
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. แสดงตารางเปรียบเทียบหลายภาพพร้อมกัน (4 Normal vs 4 Infiltration)\n",
            "ตรวจดูรูปแบบการกระจายตัวของ Heatmap ในเคสปกติหลาย ๆ คนไข้ เทียบกับเคสที่มีภาวะฝ้าแทรกซึม"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def show_batch_comparison(n_samples=4):\n",
            "    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 4.2 * n_samples))\n",
            "    \n",
            "    norm_samples = df_manifest[df_manifest['Class'] == 'Normal'].iloc[:n_samples]\n",
            "    infil_samples = df_manifest[df_manifest['Class'] == 'Infiltration'].iloc[:n_samples]\n",
            "    \n",
            "    for i in range(n_samples):\n",
            "        # 1. Normal Raw CXR\n",
            "        n_row = norm_samples.iloc[i]\n",
            "        n_img, n_cam, n_over = process_image(DATASET_DIR / n_row['Relative_Path'])\n",
            "        axes[i, 0].imshow(n_img, cmap='gray')\n",
            "        axes[i, 0].set_title(f\"Normal #{i+1}: {n_row['Image Index']}\", fontsize=10)\n",
            "        axes[i, 0].axis('off')\n",
            "        \n",
            "        # 2. Normal Overlay\n",
            "        axes[i, 1].imshow(n_over)\n",
            "        axes[i, 1].set_title(f\"Normal #{i+1} Grad-CAM\", fontsize=10)\n",
            "        axes[i, 1].axis('off')\n",
            "        \n",
            "        # 3. Infiltration Raw CXR\n",
            "        inf_row = infil_samples.iloc[i]\n",
            "        inf_img, inf_cam, inf_over = process_image(DATASET_DIR / inf_row['Relative_Path'])\n",
            "        axes[i, 2].imshow(inf_img, cmap='gray')\n",
            "        axes[i, 2].set_title(f\"Infiltration #{i+1}: {inf_row['Image Index']}\", fontsize=10)\n",
            "        axes[i, 2].axis('off')\n",
            "        \n",
            "        # 4. Infiltration Overlay + BBox\n",
            "        axes[i, 3].imshow(inf_over)\n",
            "        inf_boxes = df_bbox[df_bbox['Image Index'] == inf_row['Image Index']]\n",
            "        for _, b in inf_boxes.iterrows():\n",
            "            bx, by, bw, bh = float(b['x']), float(b['y']), float(b['w']), float(b['h'])\n",
            "            rect = patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='#00FF66', facecolor='none')\n",
            "            axes[i, 3].add_patch(rect)\n",
            "        axes[i, 3].set_title(f\"Infiltration #{i+1} + BBox\", fontsize=10)\n",
            "        axes[i, 3].axis('off')\n",
            "        \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "show_batch_comparison(n_samples=4)"
        ]
    })

    # 8. XAI-BBox alignment check (IoU / Pointing Game concept)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. การวัดความสอดคล้องระหว่าง Grad-CAM และ Ground-Truth Bounding Box (Pointing Game / XAI Alignment)\n",
            "วัดว่าตำแหน่งที่ Grad-CAM มีค่าความเข้มข้นสูงสุด (Peak Activation) ตกอยู่ภายในกรอบ Bounding Box ของรอยโรคที่แพทย์ระบุไว้หรือไม่"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "infil_records = df_manifest[df_manifest['Class'] == 'Infiltration']\n",
            "hits = 0\n",
            "total_tested = 0\n",
            "\n",
            "print(\"กำลังคำนวณการตรวจจับตำแหน่ง Peak Attention บนภาพ Infiltration 100 ภาพ...\")\n",
            "\n",
            "results_list = []\n",
            "for idx, row in infil_records.iterrows():\n",
            "    img_name = row['Image Index']\n",
            "    img_path = DATASET_DIR / row['Relative_Path']\n",
            "    boxes = df_bbox[df_bbox['Image Index'] == img_name]\n",
            "    if boxes.empty:\n",
            "        continue\n",
            "        \n",
            "    img_pil, cam, _ = process_image(img_path)\n",
            "    \n",
            "    # หาตำแหน่งพิกัด (y, x) ที่มีค่า Activation สูงสุด\n",
            "    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)\n",
            "    \n",
            "    # ตรวจสอบว่าจุด Peak อยู่ใน Bounding Box ใดๆ หรือไม่\n",
            "    is_hit = False\n",
            "    for _, b in boxes.iterrows():\n",
            "        bx, by, bw, bh = float(b['x']), float(b['y']), float(b['w']), float(b['h'])\n",
            "        if (bx <= peak_x <= bx + bw) and (by <= peak_y <= by + bh):\n",
            "            is_hit = True\n",
            "            break\n",
            "            \n",
            "    if is_hit:\n",
            "        hits += 1\n",
            "    total_tested += 1\n",
            "    results_list.append({\n",
            "        'Image Index': img_name,\n",
            "        'Peak_X': peak_x,\n",
            "        'Peak_Y': peak_y,\n",
            "        'Inside_BBox': is_hit\n",
            "    })\n",
            "\n",
            "df_xai_eval = pd.DataFrame(results_list)\n",
            "hit_rate = (hits / total_tested) * 100\n",
            "print(f\"\\nผลการประเมิน Pointing Game บนภาพ Infiltration ({total_tested} ภาพ):\")\n",
            "print(f\"- จุด Peak Activation ตกใน Bounding Box จริง: {hits}/{total_tested} ({hit_rate:.1f}%)\")\n",
            "print(\"หมายเหตุ: โมเดล Baseline Pretrained จาก ImageNet ยังไม่ได้ Fine-tune ให้เฉพาะเจาะจงกับฝ้า Infiltration จึงทำหน้าที่เป็น Base Comparator สำหรับรอบถัดไป\")\n",
            "df_xai_eval.head(10)"
        ]
    })

    # 9. Random explorer
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. ฟังก์ชันสุ่มดูคู่ภาพแบบ Interactive (Random Pair Explorer)\n",
            "รันเซลล์นี้เพื่อสุ่มดูคู่ภาพ Normal vs Infiltration คู่ใหม่ ๆ จากชุดทดลอง 200 ภาพ"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def explore_random_pair():\n",
            "    rand_norm = random.randint(0, 99)\n",
            "    rand_infil = random.randint(0, 99)\n",
            "    print(f\"สุ่มคู่ที่: Normal #{rand_norm} | Infiltration #{rand_infil}\")\n",
            "    compare_normal_vs_infil(rand_norm, rand_infil)\n",
            "\n",
            "explore_random_pair()"
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

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2, ensure_ascii=False)
        
    print(f"Generated notebook successfully at: {notebook_path}")

if __name__ == "__main__":
    build_notebook()
