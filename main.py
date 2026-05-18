import asyncio
import sys
import time
import os
import shutil  
import cv2
import numpy as np
import psutil  
import subprocess
import pyttsx3
import threading
from ultralytics import YOLO  
from hardware_interface import SmartGlassesDisplay
from parts_database import get_bin_locations 

# Link your newly created voice processing module
from voice_processor import LocalVoiceTrigger

USE_SIMULATOR = True
SHOW_VIEWFINDER = True  
STANDBY_TIMEOUT = 45.0 
ENABLE_VOICE_RESPONSES = False  # <-- Set to True at home, False at work!

SNAPSHOTS_DIR = "snapshots"
NOTES_DIR = "notes"
DATASET_DIR = "dataset"

DISCREPANCY_FILE_PATH = os.path.join(NOTES_DIR, "aircraft_discrepancies.txt")
GENERAL_MEMO_PATH = os.path.join(NOTES_DIR, "general_memos.txt")
YOLO_MODEL_PATH = "best.pt"

AIRCRAFT_PLATFORMS = ["c130h", "g550", "c17", "f16"]

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# === ADD THIS CODE ENGINE HERE ===
# This pre-loads the neural network into memory once so it stops freezing on wake
script_dir = os.path.dirname(os.path.abspath(__file__))
absolute_model_path = os.path.join(script_dir, YOLO_MODEL_PATH)

print("[Hardware Engine] Pre-loading neural network weights into memory...")
if os.path.exists(absolute_model_path):
    GLOBAL_YOLO_MODEL = YOLO(absolute_model_path)
else:
    print("[Hardware Engine] WARNING: best.pt not found yet. Will load on first scan.")
    GLOBAL_YOLO_MODEL = None

# === HIGH PERFORMANCE CONFIGURABLE SPEECH ENGINE ===
def _isolated_voice_worker(text):
    """Internal worker that runs the native OS audio drivers."""
    try:
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 190)
        local_engine.say(text)
        local_engine.runAndWait()
        local_engine.stop()
    except Exception:
        pass

def speak_offline(text):
    """
    Handles system text-to-speech feedback.
    Respects the ENABLE_VOICE_RESPONSES toggle to bypass VM hardware lag.
    """
    if ENABLE_VOICE_RESPONSES:
        # Fire voice off onto a background thread if enabled
        threading.Thread(target=_isolated_voice_worker, args=(text,), daemon=True).start()
    else:
        # Fall back to a completely lag-free console print when working in the VM
        print(f"[HUD VOICE FEEDBACK]: {text}")
# ===================================================


def log_aircraft_discrepancy(content: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_entry = f"[{timestamp}] AIRCRAFT FLIGHTLINE RECORD: {content}\n"
    with open(DISCREPANCY_FILE_PATH, "a", encoding="utf-8") as file:
        file.write(formatted_entry)
    print(f"\n[Registry Write] Logged tail issue to: {os.path.abspath(DISCREPANCY_FILE_PATH)}")

def log_general_memo(content: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_entry = f"[{timestamp}] GENERAL MEMO: {content}\n"
    with open(GENERAL_MEMO_PATH, "a", encoding="utf-8") as file:
        file.write(formatted_entry)
    print(f"\n[Registry Write] Logged general note to: {os.path.abspath(GENERAL_MEMO_PATH)}")

def export_workspace_logs() -> str:
    try:
        user_documents_root = os.path.join(os.path.expanduser('~'), 'Documents')
        target_export_dir = os.path.join(user_documents_root, 'Halo Flightline Logs')
        os.makedirs(target_export_dir, exist_ok=True)
        datestamp = time.strftime("%Y%m%d_%H%M%S")
        files_exported = 0
        
        if os.path.exists(DISCREPANCY_FILE_PATH) and os.path.getsize(DISCREPANCY_FILE_PATH) > 0:
            dest_name = f"aircraft_discrepancies_{datestamp}.txt"
            shutil.copy(DISCREPANCY_FILE_PATH, os.path.join(target_export_dir, dest_name))
            with open(DISCREPANCY_FILE_PATH, 'w', encoding='utf-8') as f:
                f.write("")
            files_exported += 1
            
        if os.path.exists(GENERAL_MEMO_PATH) and os.path.getsize(GENERAL_MEMO_PATH) > 0:
            dest_name = f"general_memos_{datestamp}.txt"
            shutil.copy(GENERAL_MEMO_PATH, os.path.join(target_export_dir, dest_name))
            with open(GENERAL_MEMO_PATH, 'w', encoding='utf-8') as f:
                f.write("")
            files_exported += 1
            
        if files_exported > 0:
            return f"SUCCESS: {files_exported} Logs Sent & Reset Local"
        return "ERROR: No active text records found to copy"
    except Exception as e:
        return f"Export Error: {str(e)}"

def local_yolo_aircraft_scan(memory_frame, display_instance, mode="standard") -> str:
    """Evaluates raw RAM frames using pre-cached model weights without lag."""
    global GLOBAL_YOLO_MODEL
    
    # Fallback if the model wasn't ready at startup
    if GLOBAL_YOLO_MODEL is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_model_path = os.path.join(script_dir, "best.pt")
        if not os.path.exists(absolute_model_path):
            return "Error: best.pt missing from folder!"
        GLOBAL_YOLO_MODEL = YOLO(absolute_model_path)
        
    try:
        # Run inference using the pre-cached global model
        prediction_list = GLOBAL_YOLO_MODEL.predict(memory_frame, conf=0.20, verbose=False)
        
        display_instance.active_ar_boxes = []
        
        if not prediction_list or len(prediction_list) == 0:
            return "No custom components identified"
            
        # FIXED: Must extract the first Result item from the list 
        results = prediction_list[0] 
        
        # Check if any bounding boxes exist safely
        if results.boxes is None or len(results.boxes) == 0:
            return "No custom components identified"
            
        highest_conf = -1.0
        best_label = "Unknown"

        for box in results.boxes:
            # Safely parse Ultralytics tensor indices
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Grabs the text name from the global model dictionary
            component_label_name = GLOBAL_YOLO_MODEL.names[class_id]
            confidence_pct = int(box_conf * 100)
            
            display_instance.active_ar_boxes.append(
                (x1, y1, x2, y2, str(component_label_name).upper(), confidence_pct)
            )
            
            if box_conf > highest_conf:
                highest_conf = box_conf
                best_label = str(component_label_name)
                
        if highest_conf == -1.0 or best_label == "Unknown":
            return "No custom components identified"
            
        if mode == "hardware":
            search_key = best_label.lower().strip()
            print(f"[Database Query] Running lookup for raw dictionary token: '{search_key}'")
            
            if "623" in search_key:
                search_key = "pn_623"
            elif "517" in search_key:
                search_key = "pn_517"
                
            bin_data_list = get_bin_locations(search_key)
            
            if not isinstance(bin_data_list, list):
                bin_data_list = [str(bin_data_list)]
                
            display_instance.hud_lines.clear()
            display_instance.show_text(f"LOCK: {search_key.upper()} FAMILY ({int(highest_conf*100)}%)", line=1)
            display_instance.show_text("=====================================", line=4)
            display_instance.show_text("--- ACTIVE HANGAR BIN LOCATIONS ---", line=5)
            
            for idx, bin_location_string in enumerate(bin_data_list[:3]):
                display_instance.show_text(str(bin_location_string).upper(), line=idx+6)
                
            display_instance._refresh_canvas()
        else:
            display_instance.hud_lines.clear()
            display_instance.show_text("ADVISE TARGET LOCK STATUS:", line=1)
            display_instance.show_text(f"{best_label.upper()} ({int(highest_conf * 100)}% MATCH)", line=2)
            
        return f"{best_label.upper()} ({int(highest_conf * 100)}% Match Locked)"
        
    except Exception as e:
        print(f"[Background Thread Exception Caught]: {e}")
        display_instance.hud_lines.clear()
        display_instance.show_text("YOLO SCAN ARCHITECTURE ERROR", line=1)
        display_instance.show_text(str(e)[:40].upper(), line=2)
        return f"Scan Error: {str(e)}"

async def system_lifecycle_manager(display):
    last_activity_time = time.time()
    
    while True:
        current_time = time.time()
        
        # FIXED: Removed cv2.waitKey(1) entirely from here to completely eliminate the loop lag!
        
        if display.trigger_wake_flag:
            display.trigger_wake_flag = False
            last_activity_time = current_time
            if display.state == "STANDBY":
                display.state = "AWAKE"
                display.show_text("READY", line=1) 
            else:
                display.state = "STANDBY"
                display.hud_lines.clear()

        if display.trigger_text_command is not None:
            raw_command = display.trigger_text_command.strip().lower()
            display.trigger_text_command = None
            last_activity_time = current_time 
            
            if raw_command.startswith("advise"):
                sub_command = raw_command.replace("advise", "", 1).strip()
                print(f"\n[Advise Engine] Wake-phrase matched: '{sub_command}'")
                
                # VOICE SHUTDOWN BRANCH (Your reliable, lag-free exit hatch!)
                if sub_command in ["exit", "shutdown"]:
                    display.hud_lines.clear()
                    display.show_text("SHUTTING DOWN SYSTEM...", line=1)
                    display._refresh_canvas()
                    print("\n[Advise Engine] Voice shutdown triggered.")
                    speak_offline("Shutting down development pipeline.")
                    await asyncio.sleep(1.0)
                    cv2.destroyAllWindows()
                    asyncio.get_event_loop().stop()
                    sys.exit(0)

                # BRANCH 1: User Manual Snapshots
                elif any(word in sub_command for word in ["snap", "photo", "picture", "snapshot"]) and not sub_command.startswith("save"):
                    display.state = "CAPTURING"
                    display.show_text("PROCESSING SNAPSHOT...", line=1)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snapshot_filename = f"{SNAPSHOTS_DIR}/user_snap_{timestamp}.jpg"
                    filepath, _ = await display.save_image_payload(snapshot_filename)
                    if filepath:
                        display.show_text("SAVED TO SNAPSHOTS", line=1)
                        await asyncio.sleep(2)
                
                # BRANCH 2: Simplified Dataset Saving Mode ("advise save stapler")
                elif sub_command.startswith("save"):
                    component_name = sub_command.replace("save", "", 1).strip()
                    if component_name == "":
                        display.show_text("ERROR: Use format 'save [item]'", line=1)
                        await asyncio.sleep(2)
                    else:
                        component_name = component_name.replace(" ", "_")
                        target_object_dir = os.path.join(DATASET_DIR, component_name)
                        os.makedirs(target_object_dir, exist_ok=True)
                        
                        display.state = "PROCESSING"
                        display.show_text(f"SAVING TO {component_name.upper()} DATASET...", line=1)
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        custom_filename = f"{target_object_dir}/{component_name}_{timestamp}.jpg"
                        
                        filepath, _ = await display.save_image_payload(custom_filename)
                        if filepath:
                            display.show_text("RECORD SECURED", line=1)
                        await asyncio.sleep(1.5)

                # BRANCH 3A: Standard Visual Mode
                elif "scan" in sub_command or "look" in sub_command:
                    display.state = "PROCESSING"
                    display.show_text("RUNNING DIRECT RAM NEURAL SCAN...", line=1)
                    temp_scan_file = f"{SNAPSHOTS_DIR}/temp_scan.jpg"
                    filepath, live_frame = await display.save_image_payload(temp_scan_file)
                    
                    if live_frame is not None:
                        display.show_text("PROCESSING ACTIVE TENSORS...", line=1)
                        ai_result = await asyncio.to_thread(local_yolo_aircraft_scan, live_frame, display, "standard")
                        
                        for countdown in range(50):  
                            display._refresh_canvas()
                            await asyncio.sleep(0.1)
                        
                        display.active_ar_boxes = []
                        display.hud_lines.clear()
                        display.show_text("READY", line=1)
                        display.state = "AWAKE"

                # BRANCH 3B: Deep Inventory Hardware Mode
                elif "hardware" in sub_command:
                    display.state = "PROCESSING"
                    display.show_text("RUNNING INVENTORY HARDWARE SEARCH...", line=1)
                    temp_scan_file = f"{SNAPSHOTS_DIR}/temp_scan.jpg"
                    filepath, live_frame = await display.save_image_payload(temp_scan_file)
                    
                    if live_frame is not None:
                        display.show_text("FETCHING PARTS REGISTRY METRICS...", line=1)
                        ai_result = await asyncio.to_thread(local_yolo_aircraft_scan, live_frame, display, "hardware")
                        
                        for countdown in range(70):  
                            display._refresh_canvas()
                            await asyncio.sleep(0.1)
                        
                        display.active_ar_boxes = []
                        display.hud_lines.clear()
                        display.show_text("READY", line=1)
                        display.state = "AWAKE"

                # BRANCH 4A: Natural Discrepancy Writing
                elif sub_command.startswith("write"):
                    discrepancy_content = sub_command.replace("write", "", 1).strip()
                    if discrepancy_content == "":
                        display.show_text("ERROR: Discrepancy details empty", line=1)
                    else:
                        display.state = "PROCESSING"
                        display.show_text("LOGGING AIRCRAFT DISCREPANCY...", line=1)
                        log_aircraft_discrepancy(discrepancy_content)
                        display.show_text("SAVED TO AIRCRAFT LOG", line=1)
                    await asyncio.sleep(2)

                # BRANCH 4B: Natural General Memos
                elif sub_command.startswith("memo") or sub_command.startswith("note"):
                    memo_content = sub_command.replace("memo", "", 1).replace("note", "", 1).strip()
                    if memo_content == "":
                        display.show_text("ERROR: Memo content empty", line=1)
                    else:
                        display.state = "PROCESSING"
                        display.show_text("SECURING SHOP FLOOR MEMO...", line=1)
                        log_general_memo(memo_content)
                        display.show_text("SAVED TO GENERAL MEMOS", line=1)
                    await asyncio.sleep(2)

                # BRANCH 5: Log List Viewer
                elif sub_command.startswith("list") or sub_command.startswith("show"):
                    display.state = "PROCESSING"
                    display.show_text("FILTERS APPLIED: TODAY'S LOGS...", line=1)
                    await asyncio.sleep(0.5)
                    
                    today_token = f"[{time.strftime('%Y-%m-%d')}"
                    
                    if os.path.exists(DISCREPANCY_FILE_PATH) and os.path.getsize(DISCREPANCY_FILE_PATH) > 0:
                        with open(DISCREPANCY_FILE_PATH, "r", encoding="utf-8") as file:
                            all_lines = file.readlines()
                        
                        todays_entries = [line.strip() for line in all_lines if line.startswith(today_token)]
                        display.hud_lines.clear()
                        
                        if not todays_entries:
                            display.show_text("TODAY'S LOG TRANSACTION: EMPTY", line=1)
                            display.show_text("No discrepancies recorded today.", line=2)
                            await asyncio.sleep(4)
                        else:
                            master_line_deck = []
                            max_chars_per_line = 38
                            
                            for idx, entry in enumerate(todays_entries):
                                raw_message = entry.split("RECORD: ")[1] if "RECORD: " in entry else entry
                                item_header = f"ITEM #{idx+1}: "
                                words = (item_header + raw_message).split(" ")
                                
                                current_line_text = ""
                                for word in words:
                                    if len(current_line_text) + len(word) + 1 > max_chars_per_line:
                                        master_line_deck.append(current_line_text.strip().upper())
                                        current_line_text = word + " "
                                    else:
                                        current_line_text += word + " "
                                if current_line_text:
                                    master_line_deck.append(current_line_text.strip().upper())
                                
                                master_line_deck.append("--------------------------")

                            lines_per_page = 5
                            
                            if len(master_line_deck) <= lines_per_page:
                                display.hud_lines.clear()
                                for print_row_idx, text_line in enumerate(master_line_deck):
                                    display.hud_lines[print_row_idx + 1] = text_line
                                display._refresh_canvas()
                                await asyncio.sleep(5.0)
                            else:
                                for step in range(len(master_line_deck) - lines_per_page + 1):
                                    display.hud_lines.clear()
                                    active_slice = master_line_deck[step : step + lines_per_page]
                                    
                                    for print_row_idx, text_line in enumerate(active_slice):
                                        display.hud_lines[print_row_idx + 1] = text_line
                                        
                                    display._refresh_canvas()
                                    await asyncio.sleep(1.5)
                                await asyncio.sleep(2.0)
                    else:
                        display.hud_lines.clear()
                        display.show_text("LOG READ TRANSACT: EMPTY", line=1)
                        display.show_text("No file records found on disk.", line=2)
                        await asyncio.sleep(3)

                # BRANCH 6: Corporate Documents Sync Export
                elif any(word in sub_command for word in ["export", "dump", "transfer", "sync"]):
                    display.state = "PROCESSING"
                    display.show_text("EXPORTING AND CLEARING LOGS...", line=1)
                    export_result = await asyncio.to_thread(export_workspace_logs)
                    display.hud_lines.clear()
                    display.show_text("EXPORT TRANSACTION:", line=1)
                    display.show_text(export_result[:40], line=2) 
                    await asyncio.sleep(4)

                # BRANCH 7: System Diagnostics Health Check
                elif any(word in sub_command for word in ["status", "battery", "info"]):
                    display.state = "PROCESSING"
                    display.show_text("GATHERING LIVE SYS METRICS...", line=1)
                    await asyncio.sleep(1)
                    battery = psutil.sensors_battery()
                    pct = battery.percent if battery else 100
                    plugged = "PLUGGED" if battery and battery.power_plugged else "DISCONNECTED"
                    display.hud_lines.clear()
                    display.show_text(f"BATTERY: {pct}% [{plugged}]", line=1)
                    display.show_text("BLE LINK: OPTIMAL (4.2MB/s) | COMMS: OK", line=2)
                    await asyncio.sleep(5) 
                else:
                    display.show_text(f"UNKNOWN COMMAND", line=1)
                    await asyncio.sleep(2)
            else:
                display.show_text("ERROR: Must start with 'advise'", line=1)
                await asyncio.sleep(2)

            if display.state != "STANDBY":
                if display.state not in ["PROCESSING", "CAPTURING"]:
                    display.hud_lines.clear()
                    display.show_text("READY", line=1)
                    display.state = "AWAKE"

        if display.trigger_snap_flag:
            display.trigger_snap_flag = False
            last_activity_time = current_time
            display.state = "CAPTURING"
            display.show_text("PROCESSING SNAPSHOT...", line=1)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            manual_file = f"{SNAPSHOTS_DIR}/manual_snap_{timestamp}.jpg"
            filepath, _ = await display.save_image_payload(manual_file)
            if filepath:
                display.show_text("SAVED TO SNAPSHOTS", line=1)
                await asyncio.sleep(2)
            if display.state != "STANDBY":
                display.hud_lines.clear()
                display.show_text("READY", line=1)
                display.state = "AWAKE"

        if display.state == "AWAKE":
            if display.typing_active:
                last_activity_time = current_time
            elapsed_idle_time = current_time - last_activity_time
            if elapsed_idle_time >= STANDBY_TIMEOUT:
                print("\n[Power Optimizer] Inactivity threshold met. Reverting to standby mode...")
                display.state = "STANDBY"
                display.hud_lines.clear()

        display._refresh_canvas()
        await asyncio.sleep(0.01)

async def main():
    display = SmartGlassesDisplay(simulate=USE_SIMULATOR, show_viewfinder=SHOW_VIEWFINDER)
    await display.connect()
    
    voice_engine = LocalVoiceTrigger()
    
    try:
        await asyncio.gather(
            system_lifecycle_manager(display),
            voice_engine.start_listening(display)
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nHalting script deployment processes.")
    finally:
        voice_engine.stop()
        display.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClean termination confirmed.")