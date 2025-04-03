import torch
import os
import sys

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

def check_model_precision(model):
    """Check model parameters precision"""
    dtypes = {}
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        if dtype_str not in dtypes:
            dtypes[dtype_str] = 0
        dtypes[dtype_str] += 1
    
    return dtypes

# Load model and check precision
print("Loading SSL model...")
ssl_model = cnhubert.get_model()

# Check initial precision
initial_precision = check_model_precision(ssl_model)
print(f"Initial model precision: {initial_precision}")

# Convert to target dtype
target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
ssl_model = ssl_model.to(target_dtype).to("cuda" if torch.cuda.is_available() else "cpu")

# Check after conversion
converted_precision = check_model_precision(ssl_model)
print(f"After conversion precision: {converted_precision}")

# Test with dummy input
print("Testing with dummy input...")
dummy_input = torch.ones(16000).to("cuda" if torch.cuda.is_available() else "cpu").to(target_dtype)
dummy_output = ssl_model.model(dummy_input.unsqueeze(0))
print(f"Output type: {dummy_output['last_hidden_state'].dtype}")
print("Test successful!")