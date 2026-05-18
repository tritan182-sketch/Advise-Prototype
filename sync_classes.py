# sync_classes.py
import os

def update_classes_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    target_labels_dir = os.path.join(script_dir, "yolotrain", "labels", "train")
    
    # Ensure the destination folder exists before writing
    os.makedirs(target_labels_dir, exist_ok=True)
    
    if not os.path.exists(dataset_dir):
        print(f"[Sync Error] The 'dataset/' folder does not exist yet at {dataset_dir}")
        return

    # 1. SCAN DIRECTORY FOR FOLDERS
    # Pulls every subfolder name inside dataset/ and ignores any lone files
    discovered_classes = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            discovered_classes.append(item.lower()) # Force lowercase for naming consistency
            
    # Sort them alphabetically so the numbering index remains stable and predictable
    discovered_classes.sort()

    if not discovered_classes:
        print("[Sync Warning] No item folders found inside your 'dataset/' folder yet.")
        return

    # 2. WRITE TO DESTINATION
    output_path = os.path.join(target_labels_dir, "classes.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for class_name in discovered_classes:
            f.write(f"{class_name}\n")
            
    print(f"\n[Sync Success] Automatically updated list from your dataset folders!")
    print(f"Destination: {output_path}")
    print(f"Active Class Index Map:")
    for index, name in enumerate(discovered_classes):
        print(f"  [{index}] -> {name.upper()}")

if __name__ == "__main__":
    update_classes_file()