# Save this as debug_ssl.py
import torch
import os
import sys
import numpy as np
import librosa

# Add path to GPT-SoVITS
gpt_sovits_path = os.path.join(os.getcwd(), 'GPT-SoVITS')
sys.path.append(gpt_sovits_path)

# Add GPT_SoVITS module directory to path
gpt_sovits_module_path = os.path.join(gpt_sovits_path, 'GPT_SoVITS')
sys.path.append(gpt_sovits_module_path)

from feature_extractor import cnhubert

# Set up environment
os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["IS_HALF"] = "True" if torch.cuda.is_available() else "False"
os.environ["PRECISION_DTYPE"] = "torch.bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "torch.float16"

# Load CNHuBERT model
cnhubert_base_path = os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
cnhubert.cnhubert_base_path = cnhubert_base_path

print("Loading SSL model...")
ssl_model = cnhubert.get_model()

# Convert model to target precision
target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Force convert all SSL model parameters 
ssl_model = ssl_model.to(device)
for param in ssl_model.parameters():
    param.data = param.data.to(target_dtype)

print(f"SSL model parameters now using {next(ssl_model.parameters()).dtype}")

# Now test with a real audio file
ref_audio = os.path.join("audio", "reference", "2b_M5171_S0030_G0050_0284.wav")
if not os.path.exists(ref_audio):
    print(f"Reference audio not found at {ref_audio}")
    # Try to find any WAV file as a fallback
    for root, dirs, files in os.walk("audio"):
        for file in files:
            if file.endswith(".wav"):
                ref_audio = os.path.join(root, file)
                print(f"Using fallback audio: {ref_audio}")
                break
        if ref_audio != "2b_M5171_S0030_G0050_0284.wav":
            break

print(f"Loading audio from {ref_audio}")
wav16k, sr = librosa.load(ref_audio, sr=16000)
print(f"Audio loaded with shape {wav16k.shape} and sr {sr}")

# Create tensor with exact same sequence as in the TTS code
wav16k_tensor = torch.from_numpy(wav16k)

# Method 1: Convert to device then dtype
wav16k_a = wav16k_tensor.to(device).to(target_dtype)
print(f"Method 1 tensor dtype: {wav16k_a.dtype}")

# Method 2: Convert to dtype then device
wav16k_b = wav16k_tensor.to(target_dtype).to(device)
print(f"Method 2 tensor dtype: {wav16k_b.dtype}")

# Test model with input
print("Testing SSL model with audio input...")
with torch.no_grad():
    try:
        output_a = ssl_model.model(wav16k_a.unsqueeze(0))
        print(f"✓ SSL model worked with Method 1 input! Output shape: {output_a['last_hidden_state'].shape}")
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
    
    try:
        output_b = ssl_model.model(wav16k_b.unsqueeze(0))
        print(f"✓ SSL model worked with Method 2 input! Output shape: {output_b['last_hidden_state'].shape}")
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")