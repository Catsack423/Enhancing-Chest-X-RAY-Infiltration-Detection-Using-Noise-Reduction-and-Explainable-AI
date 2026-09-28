import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR.parents[2] / 'data' / 'versions' / '3'
OUTPUT_DIR = CURRENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Loading data from: {DATASET_DIR}")
print(f"Saving outputs to: {OUTPUT_DIR}")

# 1. Load BBox Data
bbox_csv_path = DATASET_DIR / 'BBox_List_2017.csv'
df_bbox = pd.read_csv(bbox_csv_path).iloc[:, :6]
df_bbox.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']

# 2. Index Images
image_path_map = {}
for img_file in DATASET_DIR.glob('images_*/images/*.png'):
    image_path_map[img_file.name] = img_file

print(f"Total indexed images: {len(image_path_map)}")

# Disease Colors
DISEASE_COLORS = {
    'Atelectasis': '#FF3366',
    'Cardiomegaly': '#FF9900',
    'Effusion': '#00D4FF',
    'Infiltration': '#00FF66',
    'Infiltrate': '#00FF66',
    'Mass': '#FF00FF',
    'Nodule': '#FFFF00',
    'Pneumonia': '#FF5722',
    'Pneumothorax': '#76FF03'
}

def plot_and_save(image_name, save_filename):
    if image_name not in image_path_map:
        print(f"Image not found: {image_name}")
        return
    img = Image.open(image_path_map[image_name])
    boxes = df_bbox[df_bbox['Image Index'] == image_name]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    
    for _, row in boxes.iterrows():
        label = row['Finding Label']
        x, y, w, h = float(row['x']), float(row['y']), float(row['w']), float(row['h'])
        color = DISEASE_COLORS.get(label, '#FF0000')
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(
            x, max(0, y - 10),
            f"{label} [{int(w)}x{int(h)}]",
            color='black', fontsize=10, weight='bold',
            bbox=dict(boxstyle='square,pad=0.2', facecolor=color, edgecolor='none', alpha=0.9)
        )
        
    findings = ', '.join(boxes['Finding Label'].unique())
    ax.set_title(f"Image: {image_name} | Finding(s): {findings}", fontsize=12, pad=12, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    save_path = OUTPUT_DIR / save_filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")

# Generate Single Sample
sample_single = df_bbox.iloc[0]['Image Index']
plot_and_save(sample_single, "sample_single.png")

# Generate Multi-BBox Sample
multi_box_counts = df_bbox['Image Index'].value_counts()
multi_box_images = multi_box_counts[multi_box_counts > 1].index.tolist()
if multi_box_images:
    plot_and_save(multi_box_images[0], "sample_multi_bbox.png")

# Generate 8-Disease Grid
diseases = df_bbox['Finding Label'].unique()
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, disease in enumerate(diseases):
    sample_row = df_bbox[df_bbox['Finding Label'] == disease].iloc[0]
    img_name = sample_row['Image Index']
    img = Image.open(image_path_map[img_name])
    
    ax = axes[i]
    ax.imshow(img, cmap='gray')
    
    color = DISEASE_COLORS.get(disease, 'red')
    x, y, w, h = sample_row['x'], sample_row['y'], sample_row['w'], sample_row['h']
    
    rect = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(
        x, max(0, y - 10), disease,
        color='black', fontsize=9, weight='bold',
        bbox=dict(boxstyle='square,pad=0.2', facecolor=color, edgecolor='none', alpha=0.9)
    )
    ax.set_title(f"{disease}\n({img_name})", fontsize=11, weight='bold', pad=8)
    ax.axis('off')

plt.tight_layout()
grid_path = OUTPUT_DIR / "sample_8_diseases_grid.png"
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {grid_path.name}")

print("All sample previews generated successfully!")
