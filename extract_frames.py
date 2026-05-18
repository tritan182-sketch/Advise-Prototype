import cv2
import os

def extract_frames(video_path, output_folder, frame_interval_ms=500):
    """
    Extracts frames from a video file with unique, collision-proof prefixes.
    Targets your exact local YOLO repository directories.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[Storage Engine] Created folder: {output_folder}")

    # Extract the base name of the video file to use as a prefix
    # Example: "bracket_test.mp4" -> prefix becomes "bracket_test"
    video_filename = os.path.basename(video_path)
    video_prefix, _ = os.path.splitext(video_filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[IO Error] Could not read media source: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"[Pipeline Engine] Beginning frame separation for {video_filename}...")

    while True:
        # Jump directly to the required timestamp in the video file
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_count * frame_interval_ms)
        
        success, frame = cap.read()
        if not success:
            break
            
        # FIXED: Prefixing the image filename with the video source name
        # Generates collision-proof files like: bracket_test_frame_0024.jpg
        image_name = f"{video_prefix}_frame_{saved_count:04d}.jpg"
        image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(image_path, frame)
        
        saved_count += 1
        frame_count += 1

    cap.release()
    print(f"[Pipeline Engine] Finished! Extracted {saved_count} unique frames into '{output_folder}'.")

# === TAILORED PRODUCTION CONFIGURATION ===
if __name__ == "__main__":
    # 1. Name of the video file you just recorded
    video_file = "dataset/NAS517-5_2.avi" 
    
    # 2. Points directly to the training folder your train_model.py reads from
    destination = "yolotrain/images/train"

    # 3. 500ms extracts 2 frames per second. Increase to 1000ms if you are moving slowly.
    extract_frames(video_file, destination, frame_interval_ms=500)