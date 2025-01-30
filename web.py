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


from chat_logger import ChatLogger
chat_logger = ChatLogger()
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
gpt_sovits_path = os.path.join(current_dir, 'GPT-SoVits')
tools_path = os.path.join(gpt_sovits_path, 'tools')
gpt_sovits_module = os.path.join(gpt_sovits_path, 'GPT_SoVITS')

# Clear any existing paths we might have added
for p in list(sys.path):
    if 'GPT-SoVits' in p:
        sys.path.remove(p)

# Add paths in correct order for all imports
paths_to_add = [
    tools_path,              # For direct tool imports 
    gpt_sovits_path,        # For package-style imports
    gpt_sovits_module,      # For GPT_SoVITS module imports
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Now your imports should work
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

current_dir = os.path.dirname(os.path.abspath(__file__))
rvc_path = os.path.join(current_dir, 'rvc_cli')
sys.path.append(rvc_path)

# Rest of your configuration code stays the same
SOVITS_CONFIG = {
    "gpt_path": os.path.join(gpt_sovits_path, "GPT_weights_v2", "2B_JP6-e50.ckpt"),
    "sovits_path": os.path.join(gpt_sovits_path, "SoVITS_weights_v2", "2B_JP6_e25_s2450.pth"),
    "ref_audio": os.path.join(current_dir, "audio", "reference", "2b_calm_trimm.wav"),  
    "cnhubert_base_path": os.path.join(gpt_sovits_path, "pretrained_models", "chinese-hubert-base"),
    "bert_path": os.path.join(gpt_sovits_path, "pretrained_models", "chinese-roberta-wwm-ext-large"),
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
    import my_utils
    # Add sys.path modifications if needed
    sys.path.append(os.path.join(gpt_sovits_path, "GPT_SoVITS"))
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
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'cpu'

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of the current script's directory to sys.path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
# Initialize Flask app
app = Flask(__name__)

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
        gemini_process = subprocess.Popen(
            [sys.executable, 'gemini_cli.py'],
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

def run_tts_cli():
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Initializing GPT-SoVITS...\033[0m")
        
        def cleanup_tts():
            processes_ready['tts'].clear()
            print(f"\033[93m[{time.strftime('%H:%M:%S')}] TTS cleanup initiated\033[0m")
        
        threading.Thread(target=cleanup_tts, daemon=True)

        try:
            print(f"\033[94m[{time.strftime('%H:%M:%S')}] Loading TTS Config and Pipeline...\033[0m")
            tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
            tts_config.device = device
            tts_config.is_half = is_half
            tts_config.version = version
            tts_config.t2s_weights_path = SOVITS_CONFIG["gpt_path"]
            tts_config.vits_weights_path = SOVITS_CONFIG["sovits_path"]
            tts_config.cnhuhbert_base_path = SOVITS_CONFIG["cnhubert_base_path"]
            tts_config.bert_base_path = SOVITS_CONFIG["bert_path"]

            tts_pipeline = TTS(tts_config)
            
            if not os.path.exists(SOVITS_CONFIG["ref_audio"]):
                raise FileNotFoundError(f"Reference audio not found: {SOVITS_CONFIG['ref_audio']}")
                
            processes_ready['tts'].set()
            print(f"\033[92m[{time.strftime('%H:%M:%S')}] GPT-SoVITS initialization complete\033[0m")
            
        except Exception as e:
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to initialize GPT-SoVITS: {e}\033[0m")
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
                
                try:
                    text_chunks = text.decode().split('。' if SOVITS_CONFIG['output_language'] == 'Japanese' else '.')
                    all_audio_data = []
                    sample_rate = None
                    
                    for chunk in text_chunks:
                        if not chunk.strip():
                            continue
                            
                        inputs = {
                            "text": chunk.strip(),
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
                            "split_bucket": True,
                            "return_fragment": False,
                            "fragment_interval": float(SOVITS_CONFIG["tuning_params"]["fragment_interval"]),
                            "seed": int(SOVITS_CONFIG["tuning_params"]["seed"]),
                            "parallel_infer": True,
                            "repetition_penalty": float(SOVITS_CONFIG["tuning_params"]["repetition_penalty"])
                        }
                        
                        for sr, audio in tts_pipeline.run(inputs):
                            if sample_rate is None:
                                sample_rate = sr
                            all_audio_data.append(audio)

                    if all_audio_data:
                        import numpy as np
                        final_audio = np.concatenate(all_audio_data)
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sovits_output = os.path.join("audio", "temp", f"sovits_{timestamp}.wav")
                        os.makedirs(os.path.dirname(sovits_output), exist_ok=True)
                        
                        import soundfile as sf
                        sf.write(sovits_output, final_audio, sample_rate)
                        
                        # Process with RVC
                        rvc_queue.put(sovits_output)
                        output = os.path.join("audio", "out", f"out_{timestamp}.wav")

                        # Add timeout handling with cleanup
                        start_time = time.time()
                        rvc_success = False
                        try:
                            while not os.path.exists(output):
                                if time.time() - start_time > 25:  # Reduced timeout
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
        rvc_process = subprocess.Popen(
            [sys.executable, 'rvc_cli/rvc_inf_cli.py', 'server'],
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
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    await serve(app, config)

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