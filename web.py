from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.serving import is_running_from_reloader
import json
import subprocess
import time
import threading
import queue
import sys
import signal
import os
import torch

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
        "batch_size": 200,
        "speed_factor": 0.85,
        "fragment_interval": 0.01,
        "seed": 0,
        "repetition_penalty": 1.2,
    }
}

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

print(f"Python path: {sys.path}")

# Try importing
try:
    import my_utils
    print("Successfully imported my_utils")
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
    print("Successfully imported GPT_SoVITS modules")
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
#    'rvc': threading.Event()
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
    global response_counter
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Initializing GPT-SoVITS...\033[0m")
        
        def cleanup_tts():
            processes_ready['tts'].clear()
            print(f"\033[93m[{time.strftime('%H:%M:%S')}] TTS cleanup initiated\033[0m")
        
        # Register cleanup handler
        threading.Thread(target=cleanup_tts, daemon=True)

        # Initialize models
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
            
            # Verify reference audio exists
            if not os.path.exists(SOVITS_CONFIG["ref_audio"]):
                raise FileNotFoundError(f"Reference audio not found: {SOVITS_CONFIG['ref_audio']}")
                
            processes_ready['tts'].set()
            print(f"\033[92m[{time.strftime('%H:%M:%S')}] GPT-SoVITS initialization complete\033[0m")
            
        except Exception as e:
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to initialize GPT-SoVITS: {e}\033[0m")
            return

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join("audio", "out"), exist_ok=True)

        while not shutdown_event.is_set():
            try:
                text = tts_queue.get(timeout=1)
                if text is None:  # Shutdown signal
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
                        print("\033[95m[DEBUG WEB] Sending to TTS pipeline:", json.dumps(inputs, indent=2), "\033[0m")
                        for sr, audio in tts_pipeline.run(inputs):
                            if sample_rate is None:
                                sample_rate = sr
                            all_audio_data.append(audio)

                    if all_audio_data:
                        # Concatenate all audio chunks
                        import numpy as np
                        final_audio = np.concatenate(all_audio_data)
                        
                        # Generate unique filename with counter
                        response_counter += 1
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_filename = f"out_{timestamp}_{response_counter}.wav"
                        output_path = os.path.join("audio", "out", output_filename)
                        
                        # Save the audio
                        import soundfile as sf
                        sf.write(output_path, final_audio, sample_rate)
                        
                        # Create symlink for latest output
                        latest_output = os.path.join("audio", "out", "out.wav")
                        if os.path.exists(latest_output):
                            try:
                                os.remove(latest_output)
                            except Exception as e:
                                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Error cleaning up old audio: {e}\033[0m")

                        # Copy new output to latest
                        import shutil
                        shutil.copy2(output_path, latest_output)
                        
                        print(f"\033[92m[{time.strftime('%H:%M:%S')}] Audio saved to {output_path}\033[0m")
                        tts_completed.set()  # Signal completion to web interface
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
        
        # Wait for TTS to start
        while not processes_ready['tts'].is_set():
            time.sleep(0.1)
            
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting other processes...\033[0m")
        gemini_thread = threading.Thread(target=run_gemini_cli, daemon=True)
        gemini_thread.start()
        threads.append(gemini_thread)
        
        # Wait for all processes to be ready
        start_time = time.time()
        while not all(event.is_set() for event in processes_ready.values()):
            if time.time() - start_time > 30:  # 30 second timeout
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

if __name__ == '__main__':
    threads = init_threads()
    test_queues()
    
    # Wait a bit and check final status
    time.sleep(2)
    check_process_status()
    
    app.run(debug=False, host='0.0.0.0', port=5000)  # Note: Disabled debug mode