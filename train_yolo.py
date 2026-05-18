# train_model.py
import os
import yaml
from ultralytics import YOLO
# Detect if a GPU is available to speed up training, otherwise fall back to CPU
import torch
training_device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Hardware Engine] Training will run on: {training_device.upper() if isinstance(training_device, str) else 'NVIDIA GPU'}")

def launch_dynamic_training():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dataset_root = os.path.join(script_dir, "yolotrain").replace("\\", "/")
    
    classes_txt_path = os.path.join(target_dataset_root, "labels", "train", "classes.txt")
    
    if not os.path.exists(classes_txt_path):
        print(f"\n[Configuration Error] 'classes.txt' not found at: {classes_txt_path}")
        print("Please annotate your images on MakeSense.ai, export the YOLO zip, and drop the files here!")
        return

    # Read class definitions
    with open(classes_txt_path, "r", encoding="utf-8") as f:
        class_list = [line.strip() for line in f.readlines() if line.strip()]
        
    dynamic_names_dict = {index: name for index, name in enumerate(class_list)}
    dynamic_nc = len(class_list)
    
    print(f"\n[Dynamic Setup] Detected {dynamic_nc} unique parts from annotations.")
    
    # === COMPILER CRASH PROTECTION LAYOUT ===
    # Check if your physical validation folder has images inside it
    val_images_dir = os.path.join(target_dataset_root, "images", "val")
    has_val_images = False
    if os.path.exists(val_images_dir):
        has_val_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(val_images_dir))

    # Determine validation routing based on directory checks
    if has_val_images:
        val_route = 'images/val'
        print("[Path Engine] Dedicated validation files identified. Using images/val folder.")
    else:
        # SELF-VALIDATION ROUTE: Redirect val back to your training data pool to satisfy the compiler
        val_route = 'images/train'
        print("[Path Engine] WARNING: Validation folder empty. Redirecting val route to images/train to bypass crash.")

    # Build configuration completely in memory
    config_data = {
        'path': target_dataset_root,
        'train': 'images/train',
        'val': val_route,  # Bypasses the FileNotFoundError smoothly
        'nc': dynamic_nc,
        'names': dynamic_names_dict
    }
    
    # Write a temporary runtime configuration file
    runtime_yaml_path = os.path.join(script_dir, "aircraft_runtime.yaml")
    with open(runtime_yaml_path, 'w', encoding='utf-8') as file:
        yaml.dump(config_data, file)
        
    print(f"[Path Engine] Generated user-proof runtime configuration mapping.")

    # Execute training loops
    try:
        model = YOLO("yolov8n.pt")
        model.train(
            data=runtime_yaml_path,
            epochs=100,       
            imgsz=640,         # Lowered to 640 for faster prototyping cycles
            device=training_device, # Automatically uses GPU if your home or work PC has one
            workers=2          # Keeps CPU threading stable inside a Virtual Machine
        )
    finally:
        if os.path.exists(runtime_yaml_path):
            os.remove(runtime_yaml_path)

if __name__ == "__main__":
    launch_dynamic_training()