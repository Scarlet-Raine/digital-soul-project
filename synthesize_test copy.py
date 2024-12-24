import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# Load packages
import time
import yaml
from munch import Munch
import torch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *  # Assuming these are from StyleTTS2
from utils import *  # Assuming these are from StyleTTS2
from text_utils import TextCleaner

textclenaer = TextCleaner()

# --- Load the StyleTTS2 LJSpeech model ---

# Load the configuration file
config = yaml.safe_load(open("Models/LJSpeech/config.yml"))  # Update path if needed

# Load pretrained ASR model (if used)
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# Load pretrained F0 model (if used)
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# Load BERT model (if used)
from Utils.PLBERT.util import load_plbert  # Assuming this is from StyleTTS2
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

# Build the model
model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# Load the model checkpoint
params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')  # Update path if needed
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # Remove 'module.'
                new_state_dict[name] = v
            # Load params
            model[key].load_state_dict(new_state_dict, strict=False)
_ = [model[key].eval() for key in model]

# --- Set up the diffusion sampler ---

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # Empirical parameters
    clamp=False
)

# --- Define the inference function ---

def inference(text, noise, diffusion_steps=5, embedding_scale=1):
    # ... (rest of the inference function from the guide)

# --- Generate speech ---

    text = "This is a test sentence to generate some audio."  # Your desired text
    noise = torch.randn(1, 1, 256).to(device)  # Random noise for the diffusion process
wav = inference(text, noise, diffusion_steps=5, embedding_scale=1)  # Perform inference

# --- Save the audio ---

output_path = "output.wav"  # Your desired output path
torchaudio.save(output_path, torch.tensor(wav).unsqueeze(0), 24000)  # Save the audio