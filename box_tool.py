# box_tool.py
import cv2
import os

# DEFINE YOUR 2 TEST ITEMS HERE IN ORDER (Matches index 0 and 1)
CLASSES = ["stapler", "can"]

IMAGE_DIR = "yolotrain/images/train"
LABEL_DIR = "yolotrain/labels/train"
os.makedirs(LABEL_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print(f"Error: No pictures found inside {IMAGE_DIR}")
    exit()

current_idx = 0
drawing = False
ix, iy = -1, -1
current_boxes = []

def draw_bounding_boxes(event, x, y, flags, param):
    global ix, iy, drawing, current_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        # Standardize configuration options to let you assign classes using number keys
        print(f"Box mapped. Press [0] for {CLASSES[0].upper()} or [1] for {CLASSES[1].upper()}...")
        param['pending_box'] = (x1, y1, x2, y2)

def annotate_dataset():
    global current_idx, current_boxes
    cv2.namedWindow("Halo HUD Labeling Tool", cv2.WINDOW_NORMAL)
    param_dict = {'pending_box': None}
    cv2.setMouseCallback("Halo HUD Labeling Tool", draw_bounding_boxes, param_dict)

    # Automatically initialize your structural classes file to match Option 1 specifications
    with open(os.path.join(LABEL_DIR, "classes.txt"), "w") as cf:
        cf.write("\n".join(CLASSES) + "\n")

    while current_idx < len(image_files):
        img_name = image_files[current_idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        base_frame = cv2.imread(img_path)
        h, w, _ = base_frame.shape
        
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, txt_name)
        
        current_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                for line in lf.readlines():
                    cid, cx, cy, bw, bh = map(float, line.split())
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    current_boxes.append((int(cid), x1, y1, x2, y2))

        while True:
            display_canvas = base_frame.copy()
            for cid, x1, y1, x2, y2 in current_boxes:
                cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_canvas, CLASSES[cid].upper(), (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            status_msg = f"File {current_idx+1}/{len(image_files)}: {img_name} | [ENTER]: Next | [ESC]: Quit"
            cv2.putText(display_canvas, status_msg, (15, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            if param_dict['pending_box']:
                x1, y1, x2, y2 = param_dict['pending_box']
                cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (0, 165, 255), 1)

            cv2.imshow("Halo HUD Labeling Tool", display_canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord('0'), ord('1')] and param_dict['pending_box']:
                class_id = int(chr(key))
                x1, y1, x2, y2 = param_dict['pending_box']
                # Convert coordinate structures to YOLO normalized float parameters
                x_center = ((x1 + x2) / 2.0) / w
                y_center = ((y1 + y2) / 2.0) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                
                with open(label_path, "a") as lf:
                    lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
                current_boxes.append((class_id, x1, y1, x2, y2))
                param_dict['pending_box'] = None
                print(f" -> Saved {CLASSES[class_id].upper()} bounding box.")

            if key == 13:  # Press Enter to go to the next picture
                param_dict['pending_box'] = None
                current_idx += 1
                break
            elif key == 27:  # Press Escape to save and exit the script
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n[Annotation Engine] Dataset processing pass verified. Ready for train_model.py!")

if __name__ == "__main__":
    annotate_dataset()