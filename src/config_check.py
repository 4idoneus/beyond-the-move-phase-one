import sys
import torch
import cv2
import sgfmill

def system_status():
    print("="*60)
    print(f" Python Version: {sys.version.split()[0]}")
    print(f" PyTorch Version: {torch.__version__}")
    
    # GPU Check
    if torch.cuda.is_available():
        print(f" GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  WARNING: GPU not detected. Training will be slow on CPU.")

    # Library Checks
    print(f"OpenCV Version: {cv2.__version__}")
    print("SGFMill Imported Successfully")
    print("="*60)
    print("ENVIRONMENT: READY FOR PHASE I")

if __name__ == "__main__":
    system_status()