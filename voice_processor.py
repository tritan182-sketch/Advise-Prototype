# voice_processor.py
import speech_recognition as sr
import asyncio

class LocalVoiceTrigger:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.is_running = False

    async def start_listening(self, callback_target):
        self.is_running = True
        
        wake_words = ["advise", "advice", "ice", "eyes", "device", "allies"]
        scan_commands = ["scan", "look", "look at", "sc", "skin", "scale", "hardware"]

        with sr.Microphone() as source:
            print("[Voice Engine] Microphone hardware initialized.")
            
            while self.is_running:
                if callback_target.state == "STANDBY":
                    await asyncio.sleep(0.2)
                    continue

                if not hasattr(self, '_calibrated') or not self._calibrated:
                    print("[Voice Engine] Glasses AWAKE. Calibrating for background noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                    print("[Voice Engine] Calibration complete. Listening for commands...")
                    self._calibrated = True

                try:
                    # ADVANCED TIMING OVERRIDES: Stop the engine from closing mid-thought
                    self.recognizer.pause_threshold = 1.8         # Wait a full 1.8 seconds of complete silence
                    self.recognizer.non_speaking_duration = 1.0   # Keep the audio buffer open longer after words
                    self.recognizer.phrase_threshold = 0.5        # Minimum audio window to establish clear speech
                    
                    audio = await asyncio.to_thread(
                        self.recognizer.listen, 
                        source, 
                        timeout=4.0,            # Wait up to 4 seconds to start talking
                        phrase_time_limit=None  # No hard length constraints allowed
                    )
                except sr.WaitTimeoutError:
                    if callback_target.state == "STANDBY":
                        self._calibrated = False
                    continue
                
                try:
                    raw_text = await asyncio.to_thread(self.recognizer.recognize_google, audio)
                    spoken_phrase = raw_text.strip().lower()
                    print(f"[Raw Mic Picked Up]: '{spoken_phrase}'")
                    
                    words = spoken_phrase.split()
                    if not words:
                        continue

                    # 1. HARDWARE DATA INVENTORY FAST-TRACK
                    # Captures fuzzy variants like "advice hardware", "ice hardware", or standalone "hardware"
                    has_wake = any(w in words for w in wake_words)
                    if ("hardware" in words) or (has_wake and "hardware" in words):
                        print(" -> [Trigger Match] 'advise hardware'")
                        callback_target.trigger_text_command = "advise hardware"
                        continue

                    # 2. AR SCAN FAST-TRACK
                    has_scan = any(s in words for s in scan_commands)
                    if (has_wake and has_scan) or (spoken_phrase in ["scan", "look"]):
                        print(" -> [Trigger Match] 'advise scan'")
                        callback_target.trigger_text_command = "advise scan"
                        continue

                    # 3. FLUID LOGGING FAST-TRACK
                    if words[0] in wake_words and len(words) > 1:
                        # Normalize the spoken wake-phrase down to "advise"
                        words[0] = "advise"
                        
                        # Catch the word "right" or "ride" and convert to "write"
                        if words[1] in ["right", "ride", "white"]:
                            words[1] = "write"
                            
                        processed_command = " ".join(words)
                        print(f" -> [Vocab Cleaned]: '{processed_command}'")
                        callback_target.trigger_text_command = processed_command

                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"[Voice Processing Error]: {e}")
                
                if callback_target.state == "STANDBY":
                    self._calibrated = False
                await asyncio.sleep(0.05)

    def stop(self):
        self.is_running = False