from ultralytics import YOLO
import torch, os

# ✅ Path to your data.yaml file
data_yaml = r"C:\Users\Neetu\Virtual Impaired\datasets\mydataset\data.yaml"

# ✅ Auto-detect device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Load YOLOv8 model (small & fast)
model = YOLO("yolov8n.pt")

# ✅ Output folder
run_name = "blindassist_yolo_retry"

# ✅ Start training (with frequent checkpoint saving)
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    device=device,
    name=run_name,
    project="runs/train",
    optimizer="Adam",
    lr0=0.001,
    workers=4,
    patience=10,
    save=True,          # ensures checkpoints are saved
    save_period=5,      # save every 5 epochs
    verbose=True
)

print("\n✅ Training Completed Successfully!")
print(f"➡️ Checkpoints at: runs/train/{run_name}/weights/")
