from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.serving import is_running_from_reloader
import json
import subprocess
import time
import threading
from queue import Empty
import queue
import sys
import signal
import os
import torch
import discord
import re
from discord.ext import commands
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
import logger
from config import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from discord_bot import DiscordBot
import shutil
import time
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
try:
    nltk.download('cmudict')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    print(f"Failed to download NLTK resources: {e}")

# Get absolute path of script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Setup path to GPT-SoVITS and its subdirectories
gpt_sovits_path = os.path.join(current_dir, 'GPT-SoVITS')
sys.path.insert(0, gpt_sovits_path)  # Add to Python path

# Add GPT_SoVITS module directory to Python path
gpt_sovits_module_path = os.path.join(gpt_sovits_path, 'GPT_SoVITS')
sys.path.insert(0, gpt_sovits_module_path)

# Add tools directory to Python path
tools_path = os.path.join(gpt_sovits_path, 'tools')
sys.path.insert(0, tools_path)

# Add i18n directory to Python path
i18n_path = os.path.join(tools_path, 'i18n')
sys.path.insert(0, i18n_path)

# Now import modules with proper namespace
try:
    # First try direct imports from current sys.path
    from TTS_infer_pack.text_segmentation_method import get_method
    from inference_webui_fast import dict_language, cut_method
except ImportError:
    # If that fails, try with GPT_SoVITS namespace
    try:
        import GPT_SoVITS.TTS_infer_pack.text_segmentation_method
        import GPT_SoVITS.inference_webui_fast
        
        # Create references for easier access
        get_method = GPT_SoVITS.TTS_infer_pack.text_segmentation_method.get_method
        dict_language = GPT_SoVITS.inference_webui_fast.dict_language
        cut_method = GPT_SoVITS.inference_webui_fast.cut_method
    except ImportError as e:
        print(f"Critical import error: {e}")
        print("Python path:", sys.path)

# Import i18n
try:
    from i18n.i18n import I18nAuto, scan_language_list
except ImportError:
    try:
        from tools.i18n.i18n import I18nAuto, scan_language_list
    except ImportError as e:
        print(f"Failed to import i18n: {e}")
        print("sys.path:", sys.path)

# Set up i18n
language = os.environ.get("language", "Auto")
language = sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# Add RVC path
rvc_path = os.path.join(current_dir, 'rvc_cli')
sys.path.append(rvc_path)

# Create a helper function for imports
def safe_import(module_path, fallback_path=None, as_name=None):
    try:
        module = __import__(module_path, fromlist=[''])
        if as_name:
            globals()[as_name] = module
        return module
    except ImportError as e:
        if fallback_path:
            try:
                module = __import__(fallback_path, fromlist=[''])
                if as_name:
                    globals()[as_name] = module
                return module
            except ImportError as e2:
                print(f"Failed to import {module_path} or {fallback_path}: {e2}")
        else:
            print(f"Failed to import {module_path}: {e}")
        return None

# Rest of your configuration code stays the same
SOVITS_CONFIG = {
    "gpt_path": os.path.join(gpt_sovits_path, "GPT_weights_v2", "2B_JP6-e50.ckpt"),
    "sovits_path": os.path.join(gpt_sovits_path, "SoVITS_weights_v2", "2B_JP6_e25_s2450.pth"),
    "ref_audio": os.path.join(current_dir, "audio", "reference", "2b_calm_trimm.wav"),  
    "cnhubert_base_path": os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base"),  # Corrected path
    "bert_path": "hfl/chinese-roberta-wwm-ext-large",  # This can stay
    "ref_text": "薔薇の他にもたくさんある 百合や桜に涼らん月の涙",
    "ref_language": i18n("日文"),
    "output_language": i18n("英文"),
    "input_language": i18n("英文"),
    "how_to_cut": i18n("凑四句一切"),  # Add this
    "tuning_params": {
        "top_k": 40,
        "top_p": 1.0,
        "temperature": 0.7,
        "batch_size": 100,
        "speed_factor": 0.82,
        "fragment_interval": 0.01,
        "seed": 0,
        "repetition_penalty": 1.2,
    }
}

RVC_CONFIG = {
    "pth_path": os.path.join(current_dir, "models", "2BJP.pth"),
    "index_path": os.path.join(current_dir, "models", "2BJP.index"),
    "pitch": 0,
    "protect": 0.49,
    "filter_radius": 7,
    "clean_audio": True,
    "index_rate": 0.3,
    "volume_envelope": 0.5,
    "hop_length": 128,
    "f0_method": "rmvpe",
    "split_audio": False,
    "f0_autotune": False,
    "f0_autotune_strength": 0.7,
    "clean_strength": 0.7,
    "export_format": "WAV",
    "upscale_audio": False,
    "f0_file": None,
    "embedder_model": "contentvec"
}


processing_lock = threading.Lock()
RVC_TIMEOUT = 25  # seconds
audio_player = None

def reset_audio_pipeline():
    """Clean up audio resources and reset state"""
    global rvc_queue, audio_player
    print(f"\033[93m[{time.strftime('%H:%M:%S')}] Resetting audio pipeline...\033[0m")
    
    # Clear queues
    while not rvc_queue.empty():
        try:
            rvc_queue.get_nowait()
        except queue.Empty:
            pass
            
    # Clean temp files
    temp_dir = os.path.join("audio", "temp")
    for f in os.listdir(temp_dir):
        try:
            os.remove(os.path.join(temp_dir, f))
        except Exception as e:
            print(f"\033[93m[{time.strftime('%H:%M:%S')}] Cleanup error: {e}\033[0m")

    # Reset completion flags
    tts_completed.clear()

# Add to global variables
rvc_process = None
rvc_queue = queue.Queue()
rvc_completed = threading.Event()


how_to_cut = i18n("凑四句一切")
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get("version", "v2")

# Set all environment variables
os.environ["gpt_path"] = SOVITS_CONFIG["gpt_path"]
os.environ["sovits_path"] = SOVITS_CONFIG["sovits_path"]
os.environ["cnhubert_base_path"] = SOVITS_CONFIG["cnhubert_base_path"]
os.environ["bert_path"] = SOVITS_CONFIG["bert_path"]
os.environ["version"] = "v2"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="stft with return_complex=False is deprecated.")

# Create weight.json if needed
weight_json_path = os.path.join(gpt_sovits_path, "weight.json")
if not os.path.exists(weight_json_path):
    with open(weight_json_path, 'w', encoding="utf-8") as f:
        json.dump({
            'GPT': {'v2': SOVITS_CONFIG["gpt_path"]},
            'SoVITS': {'v2': SOVITS_CONFIG["sovits_path"]}
        }, f, indent=4)

# Try importing
try:
    # First ensure tools directory is in the path
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    
    # Now try to import my_utils from tools
    from tools.my_utils import *
    
    # Add GPT_SoVITS to the path for inference_webui_fast
    gpt_sovits_module = os.path.join(gpt_sovits_path, "GPT_SoVITS")
    if gpt_sovits_module not in sys.path:
        sys.path.append(gpt_sovits_module)
        
    from inference_webui_fast import (
    TTS, 
    TTS_Config, 
    dict_language, 
    dict_language_v1, 
    dict_language_v2,
    cut_method,  # Add this
    )
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()

import torch
# Set up precision for hardware acceleration
if torch.cuda.is_available():
    device = "cuda"
    # Check if GPU supports BF16 (Ampere or newer)
    if torch.cuda.get_device_capability()[0] >= 8:
        is_half = True
        dtype = torch.bfloat16  # Use BF16 for modern GPUs
        print(f"Using device: {device}, precision: BF16 (mixed precision)")
    else:
        is_half = True
        dtype = torch.float16  # Fall back to FP16 for older GPUs
        print(f"Using device: {device}, precision: FP16 (mixed precision)")
else:
    device = "cpu"
    is_half = False
    dtype = torch.float32
    print(f"Using device: {device}, precision: FP32")

print(f"Using device: {device}, precision: {dtype}")

# Add a simple chat logger class
class ChatLogger:
    def __init__(self):
        self.conversation_id = 0
        os.makedirs('logs', exist_ok=True)
        
    def start_new_conversation(self):
        self.conversation_id += 1
        
    def log_interaction(self, user_input, response, audio_file=None):
        try:
            with open(f'logs/conversation_{self.conversation_id}.log', 'a', encoding='utf-8') as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] USER: {user_input}\n")
                f.write(f"[{timestamp}] BOT: {response}\n")
                if audio_file:
                    f.write(f"[{timestamp}] AUDIO: {audio_file}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Logging error: {str(e)}\033[0m")


# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of the current script's directory to sys.path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
# Initialize Flask app
app = Flask(__name__)

chat_logger = ChatLogger()
# Global process variables
gemini_process = None
tts_process = None
#rvc_process = None
response_counter = 0
tts_initialized = threading.Event()
shutdown_event = threading.Event()
#rvc_completed = threading.Event()  # Add this line
tts_completed = threading.Event()


# Global queues for inter-process communication
gemini_queue = queue.Queue()
tts_queue = queue.Queue()
#rvc_queue = queue.Queue()

startup_order = threading.Event()
processes_ready = {
    'tts': threading.Event(),
    'gemini': threading.Event(),
    'rvc': threading.Event()
}

# Add a message queue to store Gemini responses
from collections import deque
last_responses = deque(maxlen=100)

# Add a callback function for Gemini responses
def handle_gemini_response(response_text):
    last_responses.append(response_text)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('Shutting down processes...')
    shutdown_event.set()
    if tts_process:
        tts_process.terminate()
    if gemini_process:
        gemini_process.terminate()
#    if rvc_process:
#        rvc_process.terminate()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Gemini CLI Handler
def run_gemini_cli():
    global gemini_process
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting Gemini process...\033[0m")
        gemini_script_path = os.path.join(current_dir, 'gemini_cli.py')
        gemini_process = subprocess.Popen(
            [sys.executable, gemini_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def monitor_stderr():
            while not shutdown_event.is_set():
                error = gemini_process.stderr.readline().strip()
                if error:
                    print(f"\033[93m[{time.strftime('%H:%M:%S')}] Gemini stderr: {error}\033[0m")
                    if "Gemini API connection successful" in error:
                        processes_ready['gemini'].set()
                        print(f"\033[92m[{time.strftime('%H:%M:%S')}] Gemini ready flag set\033[0m")
                    
        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        stderr_thread.start()
        
        startup_order.wait()

        while not shutdown_event.is_set():
            try:
                text_input = gemini_queue.get(timeout=1)
                if not text_input:
                    continue
                    
                print(f"\033[95m[{time.strftime('%H:%M:%S')}] Sending to Gemini: {text_input}\033[0m")
                gemini_process.stdin.write(f"{text_input}\n")
                gemini_process.stdin.flush()
                
                output = gemini_process.stdout.readline()
                print(f"\033[95m[{time.strftime('%H:%M:%S')}] Raw Gemini output: {output}\033[0m")
                
                if output:
                    try:
                        data = json.loads(output)
                        gemini_output = data.get('chatbot_response', '')
                        if gemini_output:
                            print(f"\033[95m[{time.strftime('%H:%M:%S')}] Gemini response: {gemini_output}\033[0m")
                            # Add the audio filename to the log
                            audio_filename = f"out_{time.strftime('%Y%m%d_%H%M%S')}_{response_counter}.wav"
                            chat_logger.log_interaction(text_input, gemini_output, audio_filename)
                            handle_gemini_response(gemini_output)
                            if processes_ready['tts'].is_set():
                                tts_queue.put(gemini_output)
                                print(f"\033[95m[{time.strftime('%H:%M:%S')}] Sent to TTS queue: {gemini_output}\033[0m")
                    except json.JSONDecodeError as e:
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Invalid JSON from Gemini: {output}\033[0m")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Gemini processing error: {str(e)}\033[0m")

    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Gemini handler error: {str(e)}\033[0m")
    finally:
        if gemini_process:
            gemini_process.terminate()


def process_text_in_chunks(text, max_chars=75):
    """Split text into chunks and process each separately"""
    # Split text into sentences first
    sentences = []
    for sent in re.split(r'([.!?])', text):
        if sent.strip():
            if sent in ['.', '!', '?']:
                # Append punctuation to the previous sentence
                if sentences:
                    sentences[-1] += sent
            else:
                sentences.append(sent)
    
    # Group sentences into chunks of appropriate size
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Make sure each chunk ends with proper punctuation
    for i in range(len(chunks)):
        if not chunks[i].rstrip().endswith(('.', '!', '?')):
            chunks[i] += '.'
            
    return chunks

def run_tts_cli():
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Initializing GPT-SoVITS...\033[0m")
        
        def cleanup_tts():
            processes_ready['tts'].clear()
            print(f"\033[93m[{time.strftime('%H:%M:%S')}] TTS cleanup initiated\033[0m")
        
        threading.Thread(target=cleanup_tts, daemon=True)

        try:
            print(f"\033[94m[{time.strftime('%H:%M:%S')}] Loading TTS Config and Pipeline...\033[0m")
            
            # Create output directories if they don't exist
            os.makedirs(os.path.join("audio", "out"), exist_ok=True)
            os.makedirs(os.path.join("audio", "temp"), exist_ok=True)
            
            # Import directly using absolute path
            sys.path.insert(0, gpt_sovits_module_path)
            
            # First, dynamically import our required modules using absolute imports
            TTS_module = safe_import('TTS_infer_pack.TTS', 'GPT_SoVITS.TTS_infer_pack.TTS')
            if not TTS_module:
                raise ImportError("Could not import TTS module")
                
            TTS = TTS_module.TTS
            TTS_Config = TTS_module.TTS_Config
            
            # Get text cleaner
            cleaner_module = safe_import('text.cleaner', 'GPT_SoVITS.text.cleaner')
            if cleaner_module:
                clean_text = cleaner_module.clean_text
            
            # Get segmentation method
            seg_module = safe_import('TTS_infer_pack.text_segmentation_method', 
                                     'GPT_SoVITS.TTS_infer_pack.text_segmentation_method')
            if seg_module:
                get_method = seg_module.get_method

            # Initialize TTS config
# Create output directories if they don't exist
            os.makedirs(os.path.join("audio", "out"), exist_ok=True)
            os.makedirs(os.path.join("audio", "temp"), exist_ok=True)
            
            # Initialize TTS config
            config_path = os.path.join(gpt_sovits_module_path, "configs", "tts_infer.yaml")
            if not os.path.exists(config_path):
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Config file not found: {config_path}\033[0m")
                return
                
            tts_config = TTS_Config(config_path)
            tts_config.device = device
            tts_config.is_half = is_half
            tts_config.version = version
            tts_config.dtype = dtype
            tts_config.t2s_weights_path = SOVITS_CONFIG["gpt_path"]
            tts_config.vits_weights_path = SOVITS_CONFIG["sovits_path"]
            
            # Set the correct paths for models
            tts_config.bert_base_path = "hfl/chinese-roberta-wwm-ext-large"  # HF model ID
            tts_config.cnhuhbert_base_path = SOVITS_CONFIG["cnhubert_base_path"]  # Corrected local path

            # Verify the path exists before proceeding
            hubert_path = SOVITS_CONFIG["cnhubert_base_path"]
            if not os.path.exists(hubert_path):
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Critical: CNHuBERT path does not exist: {hubert_path}\033[0m")
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Checking if pytorch_model.bin exists: {os.path.join(hubert_path, 'pytorch_model.bin')}\033[0m")
                return
            # Verify all model files exist for local files
            for key in ["t2s_weights_path", "vits_weights_path"]:
                path = getattr(tts_config, key)
                if not os.path.exists(path):
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Model file not found: {path}\033[0m")
                    return

            print(f"\033[92m[{time.strftime('%H:%M:%S')}] Creating TTS pipeline with config: {tts_config}\033[0m")
            
            # Import dictionary mapping for languages - get them directly from modules
            get_method = seg_module.get_method if seg_module else None
            
            # Import dict_language
            webui_module = safe_import('inference_webui_fast', 'GPT_SoVITS.inference_webui_fast')
            if webui_module:
                dict_language = webui_module.dict_language
                cut_method = webui_module.cut_method  # Get cut_method from webui_module
            else:
                dict_language = None
                cut_method = None
            
            # Create the TTS pipeline
            tts_pipeline = TTS(tts_config)

            # Set model precision
            if is_half:
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                    # Use BF16 on compatible GPUs
                    if hasattr(tts_pipeline, 't2s_model') and tts_pipeline.t2s_model:
                        tts_pipeline.t2s_model = tts_pipeline.t2s_model.to(torch.bfloat16)
                    if hasattr(tts_pipeline, 'vits_model') and tts_pipeline.vits_model:
                        tts_pipeline.vits_model = tts_pipeline.vits_model.to(torch.bfloat16)
                    print(f"\033[92m[{time.strftime('%H:%M:%S')}] Using BFloat16 precision\033[0m")
                else:
                    # Fall back to FP16
                    if hasattr(tts_pipeline, 't2s_model') and tts_pipeline.t2s_model:
                        tts_pipeline.t2s_model = tts_pipeline.t2s_model.to(torch.float16)
                    if hasattr(tts_pipeline, 'vits_model') and tts_pipeline.vits_model:
                        tts_pipeline.vits_model = tts_pipeline.vits_model.to(torch.float16)
                    print(f"\033[92m[{time.strftime('%H:%M:%S')}] Using Float16 precision\033[0m")
            if not os.path.exists(SOVITS_CONFIG["ref_audio"]):
                raise FileNotFoundError(f"Reference audio not found: {SOVITS_CONFIG['ref_audio']}")
            # Log precision information
            if hasattr(tts_pipeline, 'precision'):
                print(f"\033[92m[{time.strftime('%H:%M:%S')}] TTS Pipeline precision: {tts_pipeline.precision}\033[0m")
            for model_name in ['t2s_model', 'vits_model', 'bert_model', 'cnhuhbert_model']:
                if hasattr(tts_pipeline, model_name) and getattr(tts_pipeline, model_name) is not None:
                    model = getattr(tts_pipeline, model_name)
                    param = next(model.parameters(), None)
                    if param is not None:
                        print(f"\033[92m[{time.strftime('%H:%M:%S')}] {model_name} dtype: {param.dtype}\033[0m")
                
            processes_ready['tts'].set()
            print(f"\033[92m[{time.strftime('%H:%M:%S')}] GPT-SoVITS initialization complete\033[0m")
            
        except Exception as e:
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to initialize GPT-SoVITS: {e}\033[0m")
            import traceback
            traceback.print_exc()
            return

        os.makedirs(os.path.join("audio", "out"), exist_ok=True)

        while not shutdown_event.is_set():
            try:
                text = tts_queue.get(timeout=1)
                if text is None:
                    break
                    
                if not text:
                    continue
                
                text = text.encode('ascii', 'ignore').strip()
                if not text:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Empty text after sanitization\033[0m")
                    continue
                    
                print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing TTS: {text.decode()}\033[0m")
                
                # In web.py, in the run_tts_cli function, replace:
                try:
                    # Split text into chunks but preserve all parts
                    text_chunks = [chunk for chunk in text.decode().split('.') if chunk.strip()]
                    all_audio_data = []
                    sample_rate = None

                    # Clear cache before processing chunks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Ensure balanced sentences
                    full_text = text.decode().strip()
                    chunks = process_text_in_chunks(full_text)
                    print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing text in {len(chunks)} chunks\033[0m")
                    
                    all_audio_data = []
                    sample_rate = None

                    for chunk in chunks:
                        print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing chunk: {chunk}\033[0m")
                        
                        # Process each chunk with the same settings
                        inputs = {
                            "text": chunk,
                            "text_lang": dict_language[SOVITS_CONFIG["output_language"]],
                            "ref_audio_path": SOVITS_CONFIG["ref_audio"],
                            "aux_ref_audio_paths": [],
                            "prompt_text": SOVITS_CONFIG["ref_text"],
                            "prompt_lang": dict_language[SOVITS_CONFIG["ref_language"]],
                            "top_k": int(SOVITS_CONFIG["tuning_params"]["top_k"]),
                            "top_p": float(SOVITS_CONFIG["tuning_params"]["top_p"]),
                            "temperature": float(SOVITS_CONFIG["tuning_params"]["temperature"]),
                            "text_split_method": cut_method[how_to_cut],
                            "batch_size": int(SOVITS_CONFIG["tuning_params"]["batch_size"]),
                            "speed_factor": float(SOVITS_CONFIG["tuning_params"]["speed_factor"]),
                            "split_bucket": False,
                            "return_fragment": False,
                            "fragment_interval": float(SOVITS_CONFIG["tuning_params"]["fragment_interval"]),
                            "seed": int(SOVITS_CONFIG["tuning_params"]["seed"]),
                            "parallel_infer": False,
                            "repetition_penalty": float(SOVITS_CONFIG["tuning_params"]["repetition_penalty"])
                        }
                        
                        # Process the chunk
                        with torch.cuda.amp.autocast(enabled=is_half and torch.cuda.is_available(), dtype=dtype):
                            for sr, audio in tts_pipeline.run(inputs):
                                if sample_rate is None:
                                    sample_rate = sr
                                all_audio_data.append(audio)
                                
                        # Clear cache after each chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()    

                    if all_audio_data:
                        import numpy as np
                        final_audio = np.concatenate(all_audio_data)
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        # When processing with RVC
                        sovits_output = os.path.join("audio", "temp", f"sovits_{timestamp}.wav")
                        os.makedirs(os.path.dirname(sovits_output), exist_ok=True)
                        # After processing TTS
                        import soundfile as sf
                        with torch.no_grad():
                            # Ensure audio is float32 for saving
                            final_audio_float32 = final_audio.astype(np.float32)
                            
                            # Save with explicit format and proper scaling
                            sovits_output = os.path.abspath(os.path.join("audio", "temp", f"sovits_{timestamp}.wav"))
                            # Ensure proper scaling for soundfile - values should be between -1 and 1
                            if np.max(np.abs(final_audio_float32)) > 1.0:
                                final_audio_float32 = final_audio_float32 / 32768.0
                            sf.write(sovits_output, final_audio_float32, sample_rate, 'PCM_24')
                            
                            print(f"\033[92m[{time.strftime('%H:%M:%S')}] TTS output saved to {sovits_output}\033[0m")
                            print(f"\033[92m[{time.strftime('%H:%M:%S')}] File exists: {os.path.exists(sovits_output)}\033[0m")
                        
                        # Process with RVC
                        rvc_queue.put(sovits_output)
                        output = os.path.join("audio", "out", f"out_{timestamp}.wav")

                        # Add timeout handling with cleanup
                        start_time = time.time()
                        rvc_success = False
                        try:
                            while not os.path.exists(output):
                                if time.time() - start_time > 60:  # Increased timeout
                                    raise TimeoutError("RVC processing timeout")
                                time.sleep(0.2)  # More frequent checks
                            
                            # Validate audio file
                            if os.path.getsize(output) < 1024:
                                raise ValueError("Invalid RVC output file size")
                                
                            final_output = output
                            rvc_success = True

                        except (TimeoutError, ValueError) as e:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC Error: {str(e)} - Cleaning up...\033[0m")
                            # Cleanup failed files
                            for f in [sovits_output, output]:
                                try:
                                    if os.path.exists(f): os.remove(f)
                                except Exception as e:
                                    print(f"\033[93mCleanup error: {e}\033[0m")
                            return
                        
                        if final_output and os.path.exists(final_output):
                            print(f"\033[92m[{time.strftime('%H:%M:%S')}] Full pipeline complete: {final_output}\033[0m")
                            
                            # Create symlink for latest output
                            latest_output = os.path.join("audio", "out", "out.wav")
                            if os.path.exists(latest_output):
                                try:
                                    os.remove(latest_output)
                                except Exception as e:
                                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Error cleaning up old audio: {e}\033[0m")

                            shutil.copy2(final_output, latest_output)
                            
                            # Cleanup temp file
                            try:
                                os.remove(sovits_output)
                            except Exception as e:
                                print(f"\033[93m[{time.strftime('%H:%M:%S')}] Error cleaning temp file: {e}\033[0m")
                                
                            tts_completed.set()
                        else:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC processing failed\033[0m")
                    else:
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] No audio data generated\033[0m")
                        
                except Exception as e:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS synthesis error: {str(e)}\033[0m")
                    import traceback
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] {traceback.format_exc()}\033[0m")
                    continue

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS processing error: {str(e)}\033[0m")
                import traceback
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] {traceback.format_exc()}\033[0m")

    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS handler error: {str(e)}\033[0m")
        import traceback
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] {traceback.format_exc()}\033[0m")
    finally:
        processes_ready['tts'].clear()
        print(f"\033[93m[{time.strftime('%H:%M:%S')}] TTS process stopped\033[0m")

def run_rvc_cli():
    global rvc_process
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting RVC process...\033[0m")
        rvc_script_path = os.path.join(current_dir, 'rvc_cli', 'rvc_inf_cli.py')
        rvc_process = subprocess.Popen(
            [sys.executable, rvc_script_path, 'server'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def monitor_stderr():
            while not shutdown_event.is_set():
                error = rvc_process.stderr.readline().strip()
                if error:
                    print(f"\033[93m[{time.strftime('%H:%M:%S')}] RVC stderr: {error}\033[0m")
                    if "RVC Server Ready" in error:
                        processes_ready['rvc'].set()

        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        stderr_thread.start()

        while not shutdown_event.is_set():
            try:
                with processing_lock:
                    input_file = rvc_queue.get(timeout=1)
                    if input_file:
                        # Send config first
                        rvc_process.stdin.write(json.dumps(RVC_CONFIG) + '\n')
                        rvc_process.stdin.flush()
                        # Then send input file
                        rvc_process.stdin.write(f"{input_file}\n")
                        rvc_process.stdin.flush()
                        
                        # Add response timeout
                        output = ""
                        start_time = time.time()
                        while True:
                            if time.time() - start_time > RVC_TIMEOUT:
                                raise TimeoutError("RVC response timeout")
                            line = rvc_process.stdout.readline().strip()
                            if line:
                                output = line
                                break
                            time.sleep(0.1)
                        
                        print(f"\033[94m[{time.strftime('%H:%M:%S')}] RVC output: {output}\033[0m")

            except queue.Empty:
                continue
            except TimeoutError as te:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC Timeout: {str(te)}\033[0m")
                # Reset RVC process
                rvc_process.terminate()
                rvc_process = None
                processes_ready['rvc'].clear()
                run_rvc_cli()  # Restart RVC process
            except Exception as e:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Unexpected RVC error: {str(e)}\033[0m")
                processes_ready['rvc'].clear()
                run_rvc_cli()  # Restart RVC process

    except Exception as outer_e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Critical RVC failure: {str(outer_e)}\033[0m")
    
# Flask routes
def all_processes_ready():
    return all(event.is_set() for event in processes_ready.values())

@app.route('/status', methods=['GET'])
def check_status():
    return jsonify({
        'status': 'ready' if all_processes_ready() else 'initializing',
        'processes': {
            name: event.is_set() for name, event in processes_ready.items()
        }
    })

@app.route('/')
def serve_app():
    chat_logger.start_new_conversation()
    return send_from_directory('.', 'index.html')

@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory('assets', path)

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        with processing_lock:
            text_input = request.form.get('text')
            if not text_input:
                return jsonify({'error': 'No text provided'}), 400
                
            if not all_processes_ready():
                return jsonify({
                    'error': 'System not ready', 
                    'status': {name: event.is_set() for name, event in processes_ready.items()}
                }), 503
                
            gemini_queue.put(text_input)
            
            return jsonify({
                'status': 'success',
                'message': 'Processing started'
            })
            
    except Exception as e:
        reset_audio_pipeline()  # Cleanup on error
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_audio')
def get_audio():
    timestamp = request.args.get('t', '')  # Get timestamp from query parameter
    audio_path = os.path.join('audio', 'out', 'out.wav')
    if os.path.exists(audio_path):
        response = send_file(audio_path, mimetype='audio/wav')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return jsonify({'error': 'Audio not found'}), 404
    
# Add to clear the completion flag
@app.route('/reset_rvc', methods=['POST'])
def reset_tts():  # Rename the function too
    tts_completed.clear()
    return jsonify({'status': 'reset'})

# Modify get_response to include RVC status
@app.route('/get_response', methods=['GET'])
def get_response():
    if last_responses and tts_completed.is_set():
        return jsonify({
            'response': last_responses[-1],
            'audio_ready': True
        })
    return jsonify({
        'response': None,
        'audio_ready': False
    })
    
@app.route('/test_gemini', methods=['POST'])
def test_gemini():
    try:
        text_input = request.form.get('text', 'This is a test message.')
        print(f"\033[95m[{time.strftime('%H:%M:%S')}] Testing Gemini with: {text_input}\033[0m")
        
        # Send directly to Gemini process
        gemini_process.stdin.write(f"{text_input}\n")
        gemini_process.stdin.flush()
        
        # Wait for response with timeout
        output = gemini_process.stdout.readline()
        print(f"\033[95m[{time.strftime('%H:%M:%S')}] Gemini test response: {output}\033[0m")
        
        return jsonify({
            'status': 'success',
            'input': text_input,
            'output': output
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/update_tuning', methods=['POST'])
def update_tuning():
    try:
        params = request.get_json()
        for key, value in params.items():
            if key in SOVITS_CONFIG["tuning_params"]:
                # Input validation based on parameter type
                if key in ["top_k", "filter_radius", "pitch"]:
                    value = int(value)
                else:
                    value = float(value)
                SOVITS_CONFIG["tuning_params"][key] = value
        
        return jsonify({
            'status': 'success',
            'current_params': SOVITS_CONFIG["tuning_params"]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_tuning', methods=['GET'])
def get_tuning():
    return jsonify(SOVITS_CONFIG["tuning_params"])
    
def wait_for_tts():
    tts_thread = threading.Thread(target=run_tts_cli, daemon=True)
    tts_thread.start()
    
    start = time.time()
    while not processes_ready['tts'].is_set():
        if time.time() - start > 10:  
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS initialization timeout\033[0m")
            break
        time.sleep(0.1)
    return tts_thread

def init_threads():
    if not is_running_from_reloader():
        startup_order.clear()
        for event in processes_ready.values():
            event.clear()
            
        threads = []
        
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting TTS...\033[0m")
        tts_thread = threading.Thread(target=run_tts_cli, daemon=True)
        tts_thread.start()
        threads.append(tts_thread)
        
        while not processes_ready['tts'].is_set():
            time.sleep(0.1)
            
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting other processes...\033[0m")
        gemini_thread = threading.Thread(target=run_gemini_cli, daemon=True)
        rvc_thread = threading.Thread(target=run_rvc_cli, daemon=True)
        
        gemini_thread.start()
        rvc_thread.start()
        threads.extend([gemini_thread, rvc_thread])
        
        start_time = time.time()
        while not all(event.is_set() for event in processes_ready.values()):
            if time.time() - start_time > 45:
                print("\033[91mTimeout waiting for processes to be ready\033[0m")
                break
            time.sleep(0.1)
            
        startup_order.set()
        return threads

def test_queues():
    """Test queue functionality without sending messages"""
    print(f"\033[94m[{time.strftime('%H:%M:%S')}] Testing queue initialization...\033[0m")
    
    try:
        # Just verify queues exist
        if gemini_queue and tts_queue:  # Removed rvc_queue check
            print(f"\033[94m[{time.strftime('%H:%M:%S')}] All queues initialized successfully\033[0m")
        else:
            raise Exception("One or more queues failed to initialize")
    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Queue initialization error: {str(e)}\033[0m")

# Rest of the code remains the same
def check_process_status():
    status = {name: event.is_set() for name, event in processes_ready.items()}
    print(f"\033[94m[{time.strftime('%H:%M:%S')}] Current process status: {status}\033[0m")
    return status

async def run_flask():
    # Run Flask in a separate thread to not block the event loop
    def run_app():
        app.run(host='0.0.0.0', port=5000, debug=False)
    
    # Start Flask in a thread
    flask_thread = threading.Thread(target=run_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Keep the async function running
    while True:
        await asyncio.sleep(1)

async def run_discord():
    try:
        # Create and start Discord bot
        discord_bot = DiscordBot(gemini_queue, last_responses, tts_completed)
        await discord_bot.start()
    except Exception as e:
        logger.error("Failed to start Discord bot: %s", str(e))

async def cleanup():
    """Cleanup function for graceful shutdown"""
    logger.info("Shutting down servers...")
    shutdown_event.set()

async def run_all():
    # Initialize all the background threads first
    threads = init_threads()
    test_queues()
    time.sleep(2)
    check_process_status()
    
    try:
        # Run both Flask and Discord
        await asyncio.gather(
            run_flask(),
            run_discord()
        )
    except asyncio.CancelledError:
        await cleanup()

if __name__ == '__main__':
    try:
        # Initialize the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize background threads
        threads = init_threads()
        test_queues()
        time.sleep(2)
        check_process_status()
        
        try:
            # Run both servers
            logger.info("Starting servers...")
            loop.run_until_complete(asyncio.gather(
                run_flask(),
                run_discord()
            ))
        except Exception as e:
            logger.error(f"Server startup error: {str(e)}", exc_info=True)
            raise
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, initiating shutdown...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        try:
            # Cleanup
            logger.info("Starting cleanup...")
            shutdown_event.set()
            
            # Terminate processes
            if gemini_process:
                gemini_process.terminate()
                logger.info("Gemini process terminated")
            if tts_process:
                tts_process.terminate()
                logger.info("TTS process terminated")
            
            # Cancel all running tasks
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                
            # Close loop
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            logger.info("Event loop closed")
            
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}", exc_info=True)
        finally:
            logger.info("Shutdown complete")
            sys.exit(0)