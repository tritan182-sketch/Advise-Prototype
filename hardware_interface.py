# hardware_interface.py
import cv2
import numpy as np
import asyncio
import os
import time

class SmartGlassesDisplay:
    def __init__(self, simulate=True, show_viewfinder=False):
        self.simulate = simulate
        self.show_viewfinder = show_viewfinder
        # FIXED: Give it a safe local default value so it doesn't crash on boot!
        self.use_scifi = True 
        self.window_name = "Halo Glasses HUD Simulator"
        
        # FIXED: Explicitly restore the missing viewfinder window identifier string
        self.viewfinder_name = "Advise Camera Viewfinder (Laptop Only)"
        
        self.state = "STANDBY"  
        self.trigger_wake_flag = False
        self.trigger_snap_flag = False
        self.trigger_text_command = None
        
        self.typing_active = False
        self.text_buffer = ""

        self.help_text = "   *** VOICE SYSTEM VOCABULARY GUIDE:   'advise snapshot'   |   'advise scan'   |   'advise note tail'   |   'advise list' ***   "
        self.help_scroll_idx = 0
        self.last_scroll_time = time.time()
        
        self.active_ar_boxes = []

        if self.simulate:
            self.width = 640
            self.height = 480  
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._hidden_camera = cv2.VideoCapture(0)
            self.hud_lines = {}
            
            # FIXED: Ensure the window hook initializes if troubleshooting is turned on
            if self.show_viewfinder:
                cv2.namedWindow(self.viewfinder_name, cv2.WINDOW_NORMAL)
        else:
            self.frame_device = None

    async def connect(self):
        print(f"[Core Init] Simulation mode initialized successfully.")

    def show_text(self, text: str, line: int = 1):
        if self.simulate:
            self.hud_lines[line] = text
            self._refresh_canvas()
        else:
            print(f"[BLE TX Row {line}]: {text}")

    def _refresh_canvas(self):
        ret, frame = self._hidden_camera.read()
        current_time = time.time()
        
        # === 1. ISOLATED PROTOTYPE VIEWFINDER LAYER (FOR TROUBLESHOOTING) ===
        if self.show_viewfinder:
            if ret and self.state != "STANDBY":
                # Create a clean diagnostic copy so the raw feed doesn't lock up
                v_canvas = frame.copy()
                
                # Also project bounding box targets onto your troubleshooting screen
                for box_data in self.active_ar_boxes:
                    x1, y1, x2, y2, label, conf = box_data
                    cv2.rectangle(v_canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(v_canvas, f"{label} ({conf}%)", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
                
                cv2.putText(v_canvas, "[ PROTOTYPE CAMERA VIEW ]", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow(self.viewfinder_name, v_canvas)
            else:
                # Close the pop-up diagnostic window when system enters Standby mode
                try:
                    cv2.destroyWindow(self.viewfinder_name)
                except Exception:
                    pass

        # === 2. PRIMARY HUD GLASSES CANVAS RENDERING LAYER ===
        if self.state == "STANDBY":
            # Standby mode keeps screen background dark grey to save power
            self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.canvas[:, :] = (40, 40, 40) 
            cv2.putText(self.canvas, "[ STANDBY MODE ]", (210, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
            cv2.putText(self.canvas, "Press [TAB] to wake device", (190, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        else:
            # Check if the hardware hangar header text is active on row 5
            is_hardware_menu_active = self.hud_lines.get(5) == "--- ACTIVE HANGAR BIN LOCATIONS ---"
            
            if is_hardware_menu_active:
                # --- HIGH READABILITY CARD BACKGROUND BLOCK MODE ---
                # Drop the noisy real-world background and fill with deep slate gray
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.canvas[:, :] = (20, 20, 20) 
                
                # Draw an elegant glowing perimeter alignment frame for the digital data card
                cv2.rectangle(self.canvas, (10, 10), (self.width - 10, self.height - 10), (0, 255, 0), 2)
            else:
                # LIVE AR MODE: The HUD screen displays the real world directly
                if ret:
                    # Resize video frame array to perfectly scale onto our HUD glass resolution
                    self.canvas = cv2.resize(frame, (self.width, self.height))
                else:
                    self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

              # === DYNAMIC AUGMENTED REALITY BOX DRAWING LAYER ===
            # CONDITIONAL RENDER GATE: Only show boxes if the hardware catalog dashboard is completely offline
            if not is_hardware_menu_active:
                for box_data in self.active_ar_boxes:
                    x1, y1, x2, y2, label, conf = box_data
                    
                    if self.use_scifi:
                        # ==========================================================
                        # OPTION A: HIGH-TECH SCI-FI TARGET OVERLAYS
                        # ==========================================================
                        color_glow = (0, 255, 0)
                        thickness_glow = 2
                        bracket_len = min(25, int((x2 - x1) * 0.25))
                        
                        # Corner Brackets
                        cv2.line(self.canvas, (x1, y1), (x1 + bracket_len, y1), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x1, y1), (x1, y1 + bracket_len), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x2, y1), (x2 - bracket_len, y1), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x2, y1), (x2, y1 + bracket_len), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x1, y2), (x1 + bracket_len, y2), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x1, y2), (x1, y2 - bracket_len), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x2, y2), (x2 - bracket_len, y2), color_glow, thickness_glow)
                        cv2.line(self.canvas, (x2, y2), (x2, y2 - bracket_len), color_glow, thickness_glow)
                        
                        # Scanning Horizontal Sweep Line
                        scan_period = 1.5
                        sweep_pct = (current_time % scan_period) / scan_period
                        if sweep_pct > 0.5: sweep_pct = 1.0 - sweep_pct
                        sweep_pct *= 2.0
                        current_sweep_y = int(y1 + (y2 - y1) * sweep_pct)
                        cv2.line(self.canvas, (x1, current_sweep_y), (x2, current_sweep_y), (0, 230, 0), 1)
                        
                        # Flashing Center Crosshairs
                        if int(current_time * 5) % 2 == 0:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.line(self.canvas, (cx - 8, cy), (cx + 8, cy), color_glow, 1)
                            cv2.line(self.canvas, (cx, cy - 8), (cx, cy + 8), color_glow, 1)

                        # Floating Clean Text Tag
                        tag_text = f"TRACK LOCK: {label} ({conf}%)"
                        cv2.putText(self.canvas, tag_text, (x1, y1 - 8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_glow, 1, cv2.LINE_AA)
                    else:
                        # ==========================================================
                        # OPTION B: STANDARD LEAN BOUNDING BOXES (Your original code fallback)
                        # ==========================================================
                        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        tag_text = f"LOCK: {label} ({conf}%)"
                        cv2.rectangle(self.canvas, (x1, y1 - 25), (x1 + 180, y1), (0, 255, 0), -1)
                        cv2.putText(self.canvas, tag_text, (x1 + 5, y1 - 7), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                pass

            # === GRAPHICAL HUD DASHBOARD LAYER OVERLAY ===
            for line, text in self.hud_lines.items():
                y_pos = 50 + (line * 40)
                color = (0, 165, 255) if self.typing_active else (0, 255, 0)
                cv2.putText(self.canvas, text, (30, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            if self.typing_active:
                cv2.rectangle(self.canvas, (20, 320), (620, 380), (0, 165, 255), 1)
                cv2.putText(self.canvas, f"SPOKEN VOICE PHRASE: {self.text_buffer}_", (35, 355), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

            # === STATE-GATED HELP TICKER LAYER ===
            if self.state == "AWAKE":
                if current_time - self.last_scroll_time > 0.25:
                    self.help_scroll_idx = (self.help_scroll_idx + 1) % len(self.help_text)
                    self.last_scroll_time = current_time

                visible_help = (self.help_text[self.help_scroll_idx:] + self.help_text[:self.help_scroll_idx])[:55]
                cv2.line(self.canvas, (10, 420), (630, 420), (50, 50, 50), 1)
                cv2.putText(self.canvas, visible_help, (25, 455), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 0), 1, cv2.LINE_AA)
            else:
                pass

        cv2.imshow(self.window_name, self.canvas)
        self._handle_keystrokes()

    def _handle_keystrokes(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return  

        if key == 9:  
            self.trigger_wake_flag = True
            return
            
        if self.typing_active:
            if key == 13:  
                self.typing_active = False
                self.trigger_text_command = self.text_buffer
                self.text_buffer = ""
            elif key == 8:  
                self.text_buffer = self.text_buffer[:-1]
            elif 32 <= key <= 126:  
                self.text_buffer += chr(key)
            return

        if self.state == "AWAKE":
            if key == ord(' ') and not self.typing_active:  
                self.trigger_snap_flag = True
            elif key == ord('v') or key == ord('V'):  
                self.typing_active = True
                self.text_buffer = ""
                self.show_text("LISTENING...", line=1)
                self.hud_lines.pop(2, None) 

    async def save_image_payload(self, custom_filename=None):
        if custom_filename:
            filename = custom_filename
        else:
            filename = f"halo_snap_{int(time.time())}.jpg"
            
        for _ in range(4):
            self._hidden_camera.read()
            await asyncio.sleep(0.01)
            
        ret, frame = self._hidden_camera.read()
        if ret:
            # Resize matrix to standardize file writes
            frame_resized = cv2.resize(frame, (self.width, self.height))
            cv2.imwrite(filename, frame_resized)
            return filename, frame_resized 
        return None, None

    def close(self):
        if self.simulate:
            if hasattr(self, '_hidden_camera') and self._hidden_camera.isOpened():
                self._hidden_camera.release()
            cv2.destroyAllWindows()