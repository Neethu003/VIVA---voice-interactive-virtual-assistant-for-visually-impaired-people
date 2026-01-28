from ultralytics import YOLO
import os
from pathlib import Path

# 🔧 Paths
model = YOLO("yolo12n.pt")  # or use your best.pt once trained
dataset_dir = Path(r"C:\Users\Neetu\Virtual Impaired\datasets\mydataset")

train_img_dir = dataset_dir / "images" / "train"
label_out_dir = dataset_dir / "labels" / "train"
label_out_dir.mkdir(parents=True, exist_ok=True)

# 🧠 Auto-label unlabeled images
print("🔍 Auto-labeling images in:", train_img_dir)
for img_file in train_img_dir.glob("*.[jp][pn]g"):
    label_file = label_out_dir / (img_file.stem + ".txt")
    if not label_file.exists():  # Only label unlabeled images
        results = model.predict(source=str(img_file), save=False, save_txt=True, project=str(label_out_dir), name="", exist_ok=True)
print("✅ Auto-labeling complete!")
