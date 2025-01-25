from flask import Flask, request, jsonify
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
rvc_process = None
tts_initialized = threading.Event()
shutdown_event = threading.Event()

# Global queues for inter-process communication
gemini_queue = queue.Queue()
tts_queue = queue.Queue()
rvc_queue = queue.Queue()

startup_order = threading.Event()
processes_ready = {
    'tts': threading.Event(),
    'gemini': threading.Event(),
    'rvc': threading.Event()
}

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('Shutting down processes...')
    shutdown_event.set()
    if tts_process:
        tts_process.terminate()
    if gemini_process:
        gemini_process.terminate()
    if rvc_process:
        rvc_process.terminate()
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

# TTS CLI Handler
def run_tts_cli():
    global tts_process
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] TTS command preparation...\033[0m")
        tts_params = {
            'alpha': 0.6, 'beta': 0.85, 't': 0.85, 
            'pitch_adjust': 0, 'duration_scale': 1.1
        }
        
        cmd = [sys.executable, 'StyleTTS2/tts_cli.py',
            '--alpha', str(tts_params['alpha']),
            '--beta', str(tts_params['beta']),
            '--t', str(tts_params['t']),
            '--pitch', str(tts_params['pitch_adjust']),
            '--duration', str(tts_params['duration_scale']),
            '--device', device
        ]
        
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Running command: {' '.join(cmd)}\033[0m")

        tts_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # Changed to False for binary mode
            bufsize=1
        )

        def monitor_stderr():
            while True:
                try:
                    error = tts_process.stderr.readline()
                    if error:
                        error_text = error.decode().strip()
                        print(f"\033[93m[{time.strftime('%H:%M:%S')}] TTS stderr: {error_text}\033[0m")
                except Exception as e:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS stderr monitor error: {str(e)}\033[0m")
                    break

        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        stderr_thread.start()

        # Wait for initialization
        while not processes_ready['tts'].is_set():
            output = tts_process.stdout.readline()
            if output:
                output_text = output.decode().strip()
                print(f"\033[94m[{time.strftime('%H:%M:%S')}] TTS: {output_text}\033[0m")
                if "text_encoder loaded" in output_text:
                    processes_ready['tts'].set()
                    print(f"\033[92m[{time.strftime('%H:%M:%S')}] TTS initialization complete\033[0m")
                    break
                
            if tts_process.poll() is not None:
                raise Exception("TTS process terminated unexpectedly")

        # Main processing loop
        while not shutdown_event.is_set():
            try:
                text = tts_queue.get(timeout=1)
                if not text:
                    continue
                
                text = text.encode('ascii', 'ignore').strip()  # Encode to bytes directly
                if not text:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Empty text after sanitization\033[0m")
                    continue
                    
                print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing TTS: {text.decode()}\033[0m")
                
                # Send with proper line ending in bytes
                text_input = text + b'\n'
                try:
                    tts_process.stdin.write(text_input)
                    tts_process.stdin.flush()
                except IOError as e:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to write to TTS process: {e}\033[0m")
                    continue

                while True:
                    output = tts_process.stdout.readline()
                    if output:
                        output_text = output.decode().strip()
                        print(f"\033[94m[{time.strftime('%H:%M:%S')}] TTS output: {output_text}\033[0m")
                        if "Synthesized and saved successfully" in output_text:
                            break
                        if "Error during synthesis" in output_text:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS error detected\033[0m")
                            break

                time.sleep(2)  # Wait for file system

                expected_path = os.path.join(current_dir, "StyleTTS2", "result.wav")
                if os.path.exists(expected_path):
                    print(f"\033[92m[{time.strftime('%H:%M:%S')}] TTS output file verified at: {expected_path}\033[0m")
                    if processes_ready['rvc'].is_set():
                        rvc_queue.put("result.wav")
                        print(f"\033[95m[{time.strftime('%H:%M:%S')}] Triggered RVC processing\033[0m")
                else:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS output file not found at: {expected_path}\033[0m")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS processing error: {str(e)}\033[0m")

    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS handler error: {str(e)}\033[0m")
    finally:
        if tts_process:
            tts_process.terminate()

# RVC CLI Handler
def run_rvc_cli():
    global rvc_process
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting RVC process...\033[0m")
        processes_ready['rvc'].set()
        startup_order.wait()

        while not shutdown_event.is_set():
            try:
                input_filename = rvc_queue.get(timeout=1)
                if input_filename:
                    # Define tts_output here
                    tts_output = os.path.abspath(os.path.join(current_dir, "StyleTTS2", "result.wav"))
                    command = [
                        sys.executable,
                        os.path.join('rvc_cli', 'rvc_inf_cli.py'),
                        "infer",
                        "--input_path", tts_output,
                        "--output_path", os.path.join("audio", "out", "out.wav"),
                        "--pth_path", os.path.join("models", "2BJP.pth"),
                        "--index_path", os.path.join("models", "2BJP.index"),
                        "--pitch", "0",
                        "--protect", "0.49",
                        "--filter_radius", "7",
                        "--clean_audio", "true"
                    ]
                    
                    print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing RVC with command: {' '.join(command)}\033[0m")
                    
                    try:
                        process = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Print output and errors
                        if process.stdout:
                            print(f"\033[94m[{time.strftime('%H:%M:%S')}] RVC Output: {process.stdout}\033[0m")
                        if process.stderr:
                            print(f"\033[93m[{time.strftime('%H:%M:%S')}] RVC Stderr: {process.stderr}\033[0m")
                            
                        # Verify file was created
                        output_path = os.path.join("audio", "out", "out.wav")
                        if os.path.exists(output_path):
                            print(f"\033[92m[{time.strftime('%H:%M:%S')}] RVC output file generated: {output_path}\033[0m")
                        else:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC output file not found: {output_path}\033[0m")
                            
                    except subprocess.CalledProcessError as e:
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC process error: {e}\033[0m")
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC stdout: {e.stdout}\033[0m")
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC stderr: {e.stderr}\033[0m")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC processing error: {str(e)}\033[0m")
                import traceback
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC error traceback: {traceback.format_exc()}\033[0m")

    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC handler error: {str(e)}\033[0m")
        import traceback
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC handler traceback: {traceback.format_exc()}\033[0m")

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
def index():
    return "AI Voice Pipeline Server"

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        app.logger.debug("Received request on /process_text")
        text_input = request.form.get('text')
        app.logger.debug(f"Received text input: {text_input}")
        
        if not text_input:
            return jsonify({'error': 'No text provided'}), 400
            
        print(f"\033[95m[{time.strftime('%H:%M:%S')}] Received web request: {text_input}\033[0m")
        
        if not all_processes_ready():
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Process readiness check failed: {processes_ready}\033[0m")
            return jsonify({
                'error': 'System not ready', 
                'status': {name: event.is_set() for name, event in processes_ready.items()}
            }), 503
            
        # Start the pipeline by sending to Gemini
        print(f"\033[95m[{time.strftime('%H:%M:%S')}] Sending to Gemini queue: {text_input}\033[0m")
        gemini_queue.put(text_input)
        
        return jsonify({
            'status': 'success', 
            'message': 'Text processing started',
            'text': text_input
        })
    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Error processing request: {str(e)}\033[0m")
        return jsonify({'error': str(e)}), 500
    
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

# Initialize threads
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
        rvc_thread = threading.Thread(target=run_rvc_cli, daemon=True)
        
        gemini_thread.start()
        rvc_thread.start()
        threads.extend([gemini_thread, rvc_thread])
        
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
        if gemini_queue and tts_queue and rvc_queue:
            print(f"\033[94m[{time.strftime('%H:%M:%S')}] All queues initialized successfully\033[0m")
        else:
            raise Exception("One or more queues failed to initialize")
    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Queue initialization error: {str(e)}\033[0m")

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