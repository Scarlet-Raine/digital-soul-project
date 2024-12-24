import os
import torch
import soundfile as sf
import numpy as np

from infer.modules.vc.modules import VC
from configs.config import Config
from infer.lib.audio import load_audio
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.utils import load_hubert
from infer.modules.vc.pipeline import Pipeline
from dotenv import load_dotenv
import json
import resampy

hot_pink = "\033[95m"  
reset_color = "\033[0m" 

def load_config_json(self):
    json_config = {}
    for config_file in os.listdir(r"C:\Users\EVO\Documents\AI\RVC1006Nvidia\configs"):  # Use raw string literal
        if config_file.endswith(".json"):
            # Construct the full path using os.path.join
            with open(os.path.join(r"C:\Users\EVO\Documents\AI\RVC1006Nvidia\configs", config_file), "r") as f:  
                json_config[config_file.split(".")[0]] = json.load(f)
    return json_config

def voice_conversion(model_path, index_path, input_wav, output_wav):
    """
    Performs voice conversion with hardcoded paths, print statements, and a fixed 24k sample rate.
    """
    print(hot_pink + "Starting voice conversion..." + reset_color)
    os.environ["rmvpe_root"] = r"C:\Users\EVO\Documents\AI\RVC1006Nvidia\assets\rmvpe"
    config = Config()
    config.model_path = model_path

    print(hot_pink + "Loading RVC model..." + reset_color)
    vc = VC(config)

    # --- Adapted model loading logic ---
    vc.cpt = torch.load(model_path, map_location="cpu")
    vc.tgt_sr = vc.cpt["config"][-1]
    vc.cpt["config"][-3] = vc.cpt["weight"]["emb_g.weight"].shape[0]
    vc.if_f0 = vc.cpt.get("f0", 1)
    vc.version = vc.cpt.get("version", "v1")

    synthesizer_class = {
        ("v1", 1): SynthesizerTrnMs256NSFsid,
        ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): SynthesizerTrnMs768NSFsid,
        ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
    }

    vc.net_g = synthesizer_class.get(
        (vc.version, vc.if_f0), SynthesizerTrnMs256NSFsid
    )(*vc.cpt["config"], is_half=vc.config.is_half)

    vc.net_g.load_state_dict(vc.cpt["weight"], strict=False)
    vc.net_g.eval().to(vc.config.device)
    if vc.config.is_half:
        vc.net_g = vc.net_g.half()
    else:
        vc.net_g = vc.net_g.float()

    vc.pipeline = Pipeline(vc.tgt_sr, vc.config)  # Initialize pipeline here
    # --- End of adapted model loading logic ---

    print(hot_pink + "Loading RVC model..." + reset_color)

    # --- Code from infer.web.py ---
    audio = load_audio(input_wav, 22050)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]

    if vc.hubert_model is None:
        vc.hubert_model = load_hubert(vc.config)

    index_path = index_path.replace("trained", "added")

    audio_opt = vc.pipeline.pipeline(
        vc.hubert_model,
        vc.net_g,
        2,  # Hardcoded sid to 0
        audio,
        input_wav,
        times,
        0,  # Updated f0_up_key to 12 (raise by one octave)
        "rmvpe",
        index_path,
        0.4,
        vc.if_f0,
        7,
        vc.tgt_sr,
        24000,
        0.25,
        vc.version,
        0.49,
        None,
    )
    # --- End of code from infer.web.py ---

    print(hot_pink + "Saving output audio..." + reset_color)

    audio = np.array(audio, dtype=np.float32)
    sf.write(output_wav, audio, samplerate=vc.tgt_sr)

    print(hot_pink + "Voice conversion completed!" + reset_color)

if __name__ == "__main__":
    voice_conversion(
        "C:\\Users\\EVO\\Documents\\AI\\RVC1006Nvidia\\models\\2BJP.pth", 
        "C:\\Users\\EVO\\Documents\\AI\\RVC1006Nvidia\\models\\2BJP.index",
        "C:\\Users\\EVO\\Documents\\AI\\StyleTTS2\\result.wav", 
        "C:\\Users\\EVO\\Documents\\AI\\audio\\out.wav" 
    )