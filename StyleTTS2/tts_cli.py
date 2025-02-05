from cached_path import cached_path

from dp.phonemizer import Phonemizer
print("NLTK")
import nltk
nltk.download('punkt')
print("SCIPY")
from scipy.io.wavfile import write
print("TORCH STUFF")
import torch
print("START")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


import numpy as np


# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import phonemizer
from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()
import sys
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
import os

import torch
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'cpu'
global_device = device

def get_device():
    return device

# Modify model loading to explicitly use the device



to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4  

global_ref_s = None

global_ref_s = None

def get_ref_s():
    global global_ref_s
    if global_ref_s is None:
        print("Computing style for first time...")
        with torch.no_grad():
            global_ref_s = compute_style('./voice/voice_2b_short.wav')
        print("Style computed and cached.")
    else:
        print("Using cached style.")
    return global_ref_s


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor  


def compute_style(path):
    global device  # Add this
    global global_device  # Add this
    print(f"Computing style for {path}...")
    wave, sr = librosa.load(path, sr=24000)
    print(f"Loaded audio with shape {wave.shape} and sample rate {sr}")
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(global_device)
    
    with torch.no_grad():
        get_ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    
    return torch.cat([get_ref_s, ref_p], dim=1)

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))



config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config) # type: ignore

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path) # type: ignore

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert) # type: ignore
_ = [model[key].eval() for key in model]
_ = [model[key].to(get_device()) for key in model]

params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu', weights_only=True)
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

                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)  

#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# Ensure the correct arguments are passed based on the DiffusionSampler's expected inputs
sampler = DiffusionSampler(
    diffusion=model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    clamp=False
)

if hasattr(sampler, 'to'):
    sampler = sampler.to(device)

def save_audio(wav_data, sample_rate=24000):
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.wav')
        write(output_path, sample_rate, wav_data)
        print(f"Audio saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def inference(text, noise, diffusion_steps=5, embedding_scale=1):
    global device
    global global_device
    print(f"DEBUG: Starting inference with device={device}, global_device={global_device}")

    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    print("DEBUG: Text processed")

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    print("DEBUG: About to convert tokens")
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    print("DEBUG: Tokens converted and moved to device")

    with torch.no_grad():
        print("DEBUG: Starting no_grad section")
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        print("DEBUG: input_lengths created")
        text_mask = length_to_mask(input_lengths).to(device)
        print("DEBUG: text_mask created")

        print("DEBUG: About to run text_encoder")
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        print("DEBUG: text_encoder complete")

        print("DEBUG: About to run BERT")
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        print("DEBUG: BERT complete")

        print("DEBUG: About to run bert_encoder")
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        print("DEBUG: bert_encoder complete")

        print("DEBUG: About to run sampler")
        noise = noise.to(device)  # Ensure noise is on correct device
        ref_s = get_ref_s().to(device)  # Get and move reference style to device
        try:
            print("DEBUG: Starting sampler call")
            print(f"DEBUG: Device states - noise: {noise.device}, bert_dur: {bert_dur[0].device}, ref_s: {ref_s.device}")

            s_pred = sampler(
                noise=noise,
                embedding=bert_dur[0].unsqueeze(0),
                features=ref_s,
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale
            )
            print("DEBUG: Sampler call complete")
            s_pred = s_pred.squeeze(0)
        except Exception as e:
            print(f"DEBUG: Sampler error - {str(e)}")
            print(f"DEBUG: Sampler type - {type(sampler)}")
            raise

        print("DEBUG: Processing style")
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        print("DEBUG: Style processed")

        print("DEBUG: About to run predictor text_encoder")
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        print("DEBUG: predictor text_encoder complete")

        print("DEBUG: About to run predictor lstm")
        x, _ = model.predictor.lstm(d)
        print("DEBUG: predictor lstm complete")

        print("DEBUG: About to get duration")
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        print("DEBUG: duration complete")

        print("DEBUG: About to create alignment")
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        print("DEBUG: zeros created")

        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        print("DEBUG: alignment complete")

        print("DEBUG: About to encode prosody")
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        print("DEBUG: prosody encoded")
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        if save_audio(out.squeeze().cpu().numpy()):
            print("DEBUG: Audio saved successfully")
        else:
            print("DEBUG: Audio saving failed")

    return out.squeeze().cpu().numpy()



def LFinference(text, s_prev, get_ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
  global device  # Add this line
  global global_device  # And this line
  text = text.strip()
  ps = phonemizer([text], lang='en_us')
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)
  ps = ps.replace('``', '"')
  ps = ps.replace("''", '"')

  tokens = textclenaer(ps)
  tokens.insert(0, 0)
  tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
      text_mask = length_to_mask(input_lengths).to(device)

      t_en = model.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

      s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                        embedding=bert_dur,
                                        embedding_scale=embedding_scale,
                                          features=get_ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = t * s_prev + (1 - t) * s_pred

      s = s_pred[:, 128:]
      ref = s_pred[:, :128]

      ref = alpha * ref + (1 - alpha)  * get_ref_s[:, :128]
      s = beta * s + (1 - beta)  * get_ref_s[:, 128:]

      s_pred = torch.cat([ref, s], dim=-1)

      d = model.predictor.text_encoder(d_en,
                                        s, input_lengths, text_mask)

      x, _ = model.predictor.lstm(d)
      duration = model.predictor.duration_proj(x)

      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)


      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
      if model_params.decoder.type == "hifigan":
          asr_new = torch.zeros_like(en)
          asr_new[:, :, 0] = en[:, :, 0]
          asr_new[:, :, 1:] = en[:, :, 0:-1]
          en = asr_new

      F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

      asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
      if model_params.decoder.type == "hifigan":
          asr_new = torch.zeros_like(asr)
          asr_new[:, :, 0] = asr[:, :, 0]
          asr_new[:, :, 1:] = asr[:, :, 0:-1]
          asr = asr_new

      out = model.decoder(asr,
                              F0_pred, N_pred, ref.squeeze().unsqueeze(0))


  return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

def STinference(text, get_ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    global device  # Add this line
    global global_device  # And this line
    text = text.strip()
    ps = phonemizer([text], lang='en_us')
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = phonemizer([ref_text], lang='en_us')
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)


    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=get_ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * get_ref_s[:, :128]
        s = beta * s + (1 - beta)  * get_ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

def process_text(text, inference_params):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    try:
        # Sanitize the input text
        text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII chars
        text = text.strip()
        if not text:
            print("Empty text after sanitization")
            return False

        # Add debug prints
        print(f"Processing text (length {len(text)}): {text}")
        
        # Create noise tensor here to ensure device placement
        noise = torch.randn((1, 256)).unsqueeze(1).to(device)
        
        wav = inference(text, noise, diffusion_steps=20, embedding_scale=2)
        if wav is not None:
            if save_audio(wav):
                print("Synthesized and saved successfully")
                return True
            else:
                print("Failed to save audio")
        return False
    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.85)
    parser.add_argument('--t', type=float, default=0.85)
    parser.add_argument('--pitch', type=float, default=0)
    parser.add_argument('--duration', type=float, default=1.1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    global_device = device

    inference_params = {
        'alpha': args.alpha,
        'beta': args.beta,
        't': args.t,
        'pitch_adjust': args.pitch,
        'duration_scale': args.duration
    }

    while True:
        try:
            text = input("Enter text to synthesize: ")
            if not text:
                continue
            print(f"Synthesizing: {text}")
            process_text(text, inference_params)
        except KeyboardInterrupt:
            print("Exiting...")
            break
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


with torch.no_grad():
    # Add the model files to the safe globals list using a context manager
    with torch.serialization.safe_globals([
        "Models/LibriTTS/epochs_2nd_00020.pth",
        "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt"
    ]):

        print("Style computed.")

while True:
    try:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        # Get text input from the command line arguments
        if len(sys.argv) > 1:
            text = sys.argv[1]
        else:
            text = input("Enter text to synthesize: ")
        print(f"Synthesizing: {text}")
        
        start = time.time()
        wav = inference(text, get_ref_s, diffusion_steps=20, embedding_scale=2)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")

        if wav is not None:
            if save_audio(wav):
                print("Synthesized and saved successfully")
            else:
                print("Failed to save audio")
        else:
            print("No audio data generated")

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error during synthesis: {e}")
        break