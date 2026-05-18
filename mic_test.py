# mic_test.py
import sounddevice as sd

print("=== AVAILABLE AUDIO INPUT DEVICES ===")
# This pulls the official list of hardware items detected by your system
print(sd.query_devices())

print("\n=== STARTING 3-SECOND AUDIO TEST ===")
print("Please speak, snap your fingers, or make noise now...")

try:
    # Records a brief raw audio snippet at standard voice frequency
    recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
    sd.wait()  # Blocks execution until the hardware buffers finish filling
    
    print("=== RECORDING COMPLETE ===")
    print(f"Captured array data shape: {recording.shape}")
    print(f"Peak recorded audio volume: {recording.max():.4f}")
    
    if recording.max() > 0.001:
        print("\n SUCCESS: Your microphone is working and sending data to Python!")
    else:
        print("\n WARNING: Device recorded, but data is silent. Check system mute levels.")

except Exception as e:
    print(f"\n HARDWARE ERROR: Could not access microphone. Details: {e}")