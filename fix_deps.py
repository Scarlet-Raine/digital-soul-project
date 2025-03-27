import subprocess
import sys

def fix_dependencies():
    print("Fixing dependencies...")
    
    # Uninstall current torch and torchvision
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    
    # Install latest versions with CUDA 12.4 support
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ])
    
    # Update transformers
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
    
    # Verify CUDA availability
    import torch
    if torch.cuda.is_available():
        print(f"\nCUDA is available!")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWarning: CUDA is not available!")

if __name__ == "__main__":
    fix_dependencies()