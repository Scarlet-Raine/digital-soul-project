__all__ = ['compute_style', 'inference', 'LFinference', 'initialize_models', 'process_text', 'analyze_pitch', 'compute_pitch_shift']
from cached_path import cached_path
from dp.phonemizer import Phonemizer
import nltk
nltk.download('punkt')
from scipy.io.wavfile import write
import torch
import os
import librosa
import numpy as np
from nltk.tokenize import word_tokenize
import yaml
from munch import Munch
import torch.nn.functional as F
import torchaudio

# Keep the models import
from models import *
from utils import *
from text_utils import TextCleaner

# Add these back
import phonemizer.backend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.separator import Separator
import random
import os
import time
import sys

textclenaer = TextCleaner()


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4  

def initialize_phonemizer():
    global phonemizer
    phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us',
        preserve_punctuation=True,
        with_stress=True,
        separator=Separator(word=' ', syllable='|', phone=' ')
    )
    return phonemizer

def recursive_munch(d):
    """Convert dict to Munch recursively"""
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(x) for x in d]
    else:
        return d

def initialize_models():
    """Initialize all StyleTTS2 models"""
    global model, sampler, device, global_phonemizer, phonemizer, model_params

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        print("MPS would be available but cannot be used rn")

    # Initialize phonemizer with language parameter
    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        preserve_punctuation=True,
        with_stress=True,
        language='en-us',  
        separator=Separator(word=' ', syllable='|', phone=' ')
    )

    # Set up the phonemizer with the checkpoint
    phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))

    config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu', weights_only=True)
    params = params_whole['net']

    for key in model:
        if key in params:
            print(f'{key} loaded')
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    _ = [model[key].eval() for key in model]

    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
    global sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

def process_text(text):
    """Process text for TTS with phoneme conversion"""
    try:
        text = text.strip()
        text = text.replace('"', '')
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        text = ''.join(char for char in text if ord(char) < 0x10000)
        
        ps = global_phonemizer.phonemize([text])  # Remove language parameter
        if isinstance(ps, list):
            ps = ps[0]
        ps = word_tokenize(ps)
        ps = ' '.join(ps)
        
        print(f"Phonemized text: {ps}")
        return ps
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

# Keep all your existing helper functions
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, segment_length=3.0, sr=24000):
    """Compute style with enhanced prosody capture"""
    print(f"Computing style for {path}...")
    wave, sr = librosa.load(path, sr=sr)
    
    # Split into smaller segments to capture more varied prosody
    segment_samples = int(segment_length * sr)
    segments = []
    
    # Use overlapping windows to better capture transitions
    hop_length = segment_samples // 2
    for i in range(0, len(wave) - segment_samples + 1, hop_length):
        segment = wave[i:i + segment_samples]
        if len(segment) >= sr:  # Only use segments of at least 1 second
            segments.append(segment)
    
    if not segments:
        segments = [wave]
    
    # Process each segment
    all_styles = []
    for segment in segments:
        audio, _ = librosa.effects.trim(segment, top_db=30)
        if len(audio) < sr:
            continue
            
        mel_tensor = preprocess(audio).to(device)
        
        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
            combined = torch.cat([ref_s, ref_p], dim=1)
            all_styles.append(combined)
    
    if all_styles:
        styles_tensor = torch.stack(all_styles)
        weights = torch.var(styles_tensor, dim=2).mean(dim=1)
        weights = F.softmax(weights, dim=0)
        return (styles_tensor * weights.unsqueeze(1).unsqueeze(2)).sum(dim=0)
    else:
        raise ValueError("No valid segments found in audio file")

def process_text(text, lang='ja'):
    """Enhanced text processing with language handling"""
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        text = ''.join(char for char in text if ord(char) < 0x10000)
        
        # Use language-specific phonemizer
        ps = global_phonemizer.phonemize([text], language=lang)
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        
        return ps
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

def save_audio(wav_data, sample_rate=24000):
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.wav')
        write(output_path, sample_rate, wav_data)
        print(f"Audio saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def inference(text, ref_s, diffusion_steps=5, embedding_scale=1, pitch_adjust=0, duration_scale=1.0):
    """Base inference function with pitch adjustment"""
    processed_text = process_text(text)
    if not processed_text:
        return None

    tokens = textclenaer(processed_text)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        noise = torch.randn((1, 256)).unsqueeze(1).to(device)
        s_pred = sampler(noise, embedding=bert_dur[0].unsqueeze(0), features=ref_s, 
                        num_steps=diffusion_steps, embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        
        # Apply duration scaling
        duration = duration * duration_scale
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        
        # Apply pitch adjustment
        if pitch_adjust != 0:
            F0_pred = F0_pred * (2 ** (pitch_adjust / 12))
        
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                          F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()

def LFinference(text, s_prev, ref_s, alpha=0.3, beta=0.7, t=0.7, 
                diffusion_steps=5, embedding_scale=1, pitch_adjust=0, duration_scale=1.0):
    """Style-mixed inference with pitch adjustment"""
    text = text.strip()
    ps = phonemizer([text])  # Remove language parameter here too
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

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=embedding_scale,
                        features=ref_s,
                        num_steps=diffusion_steps).squeeze(1)

        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        # Apply duration scaling
        duration = duration * duration_scale
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
        
        # Apply pitch adjustment
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        if pitch_adjust != 0:
            F0_pred = F0_pred * (2 ** (pitch_adjust / 12))  # Convert semitones to frequency ratio
        
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
    return out.squeeze().cpu().numpy()[..., :-100], s_pred

def STinference(text, ref_s, ref_text, alpha=0.3, beta=0.7, 
                diffusion_steps=5, embedding_scale=1, pitch_adjust=0, duration_scale=1.0):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
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
        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=embedding_scale,
                        features=ref_s,
                        num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        # Apply duration scaling
        duration = duration * duration_scale
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
        
        # Apply pitch adjustment
        if pitch_adjust != 0:
            F0_pred = F0_pred * (2 ** (pitch_adjust / 12))

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]# weird pulse at the end of the model, need to be fixed later

def analyze_pitch(audio_path, sr=24000):
    """Analyze the average pitch of an audio file"""
    y, sr = librosa.load(audio_path, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get pitches with highest magnitude at each frame
    valid_pitches = []
    for t in range(pitches.shape[1]):
        max_mag_idx = magnitudes[:, t].argmax()
        pitch = pitches[max_mag_idx, t]
        if pitch > 0:  # Filter out zero pitches
            valid_pitches.append(pitch)
            
    if valid_pitches:
        return np.mean(valid_pitches)
    return None

def compute_pitch_shift(source_path, target_path, sr=24000):
    """Compute required pitch shift between source and target audio"""
    source_pitch = analyze_pitch(source_path, sr)
    target_pitch = analyze_pitch(target_path, sr)
    
    if source_pitch and target_pitch:
        # Convert frequency difference to semitones
        pitch_shift = 12 * np.log2(target_pitch / source_pitch)
        return -pitch_shift  # Negative because we want to shift back to source pitch
    return 0

if __name__ == "__main__":
    print("Time to synthesize!")
    
    # Initialize models
    initialize_models()
    
    # Load reference style
    with torch.no_grad():
        ref_s = compute_style('./voice/voice.wav')
    
    print("Style computed.")

    while True:
        try:
            # Get text input from the command line arguments
            if len(sys.argv) > 1:
                text = sys.argv[1]
            else:
                text = input("Enter text to synthesize: ")
            print(f"Synthesizing: {text}")
            
            start = time.time()
            wav = inference(text, ref_s, diffusion_steps=20, embedding_scale=2)
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