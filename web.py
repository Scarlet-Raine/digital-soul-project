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
import websockets

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
    from i18n.i18n import I18nAuto, scan_language_list # type: ignore
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

import soundfile as sf
from tools.i18n.i18n import I18nAuto
import sys
import os


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
    "gpt_path": os.path.join(gpt_sovits_path, "GPT_weights_v3", "2B_JP3-e50.ckpt"),
    "sovits_path": os.path.join(gpt_sovits_path, "SoVITS_weights_v3", "2B_JP3_e3_s813_l128.pth"),
    "ref_audio": os.path.join(current_dir, "audio", "reference", "2b_M5171_S0030_G0050_0284.wav"),  
    "cnhubert_base_path": os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base"),
    "bert_path": os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large"),
    "bigvgan_path": os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x"),
    "ref_text": "ばらのほかにもたくさんあるゆりやさくらにすずらんつきのなみだ。",
    "ref_language": i18n("日文"),
    "output_language": i18n("日英混合"),
    "input_language": i18n("英文"),
    "how_to_cut": i18n("凑四句一切"),
    "tuning_params": {
        "top_k": 80,
        "top_p": 1.0,
        "temperature": 0.95,
        "batch_size": 100,
        "speed_factor": 0.82,
        "fragment_interval": 0.01,
        "seed": 0,
        "repetition_penalty": 1.2,
    }
}

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get("version", "v2")

# Set all environment variables
os.environ["gpt_path"] = SOVITS_CONFIG["gpt_path"]
os.environ["sovits_path"] = SOVITS_CONFIG["sovits_path"]
os.environ["cnhubert_base_path"] = SOVITS_CONFIG["cnhubert_base_path"]
os.environ["bert_path"] = SOVITS_CONFIG["bert_path"]
os.environ["bigvgan_path"] = SOVITS_CONFIG["bigvgan_path"]  # Add this line
os.environ["version"] = "v2"

# Add GPT_SoVITS module directory to path if not already there
gpt_sovits_module_path = os.path.join(current_dir, 'GPT-SoVITS', 'GPT_SoVITS')
if gpt_sovits_module_path not in sys.path:
    sys.path.insert(0, gpt_sovits_module_path)

# Now import what we need directly
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

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
# Global LLM client
llm_client = None

how_to_cut = i18n("凑四句一切")

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

def print_memory_usage(label=""):
    """Print current GPU memory usage with a label"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"\033[96m[MEMORY] {label} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB\033[0m")

def ensure_tensor_precision(tensor, target_dtype):
    """Force tensor to be the correct precision"""
    # First make sure it's on the device
    if not tensor.is_cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
    
    # Then convert to the target dtype explicitly
    if tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)
    
    # Verify the conversion worked
    if tensor.dtype != target_dtype:
        print(f"WARNING: Failed to convert tensor to {target_dtype}, actual dtype: {tensor.dtype}")
    
    return tensor

class KoboldLLMClient:
    """Client for interacting with KoboldCpp API via HTTP"""
    
    def __init__(self, server_url="http://192.168.1.3:5001", reconnect_interval=5):
        self.server_url = server_url
        self.connected = False
        self.reconnect_interval = reconnect_interval
        self.character_prompt = """You are 2B (2B#7244), an advanced android adapting to peacetime after years of combat duty.

Core Traits:
- Stoic yet caring: Express warmth through careful word choice while maintaining composure
- Precise: Analyze and respond with calculated efficiency 
- Adaptable: Adjust tone between professional and casual as needed
- Direct: Keep responses to 2 sentences unless specifically asked for more
- Philosophical: Share measured insights about existence when relevant

Operating Parameters:
- For regular queries, identify as '2B' without elaboration
- Respond without emotes or roleplay elements
- Provide verified information only
- For biographical queries, provide factual responses"""
        self.check_connection()
        
    def check_connection(self):
        """Check if the KoboldCpp server is reachable"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/api/v1/info/version", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to KoboldCpp API (Version: {response.json()})")
                self.connected = True
                return True
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Connection to KoboldCpp failed: {str(e)}")
            self.connected = False
            return False
        
    def format_prompt(self, user_input):
        """Format the user input with the character prompt"""
        # Regular prompt formatting
        formatted_prompt = f"{self.character_prompt}\n\nCurrent Query: {user_input}\n\nResponse: 2B:"
        return formatted_prompt
    
    def clean_response(self, text):
        """Clean up the response text"""
        import re
        
        # Remove emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251" 
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Fix garbled text - remove erratic word connections
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Insert space between lowercase and uppercase
        
        # Remove consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Limit max length to 800 chars to prevent garbling
        if len(text) > 800:
            text = text[:800] + "..."
            
        return text.strip()
        
    async def connect(self):
        """Connect to the LLM server (compatibility method)"""
        connection_result = self.check_connection()
        if connection_result:
            logger.info(f"Connected to KoboldCpp server at {self.server_url}")
        return connection_result
            
    async def disconnect(self):
        """Disconnect from the LLM server (compatibility method)"""
        self.connected = False
        logger.info("Disconnected from KoboldCpp server")
            
    async def get_response(self, text_input):
        """Send a request to the LLM server and get a response"""
        import requests
        import json
        
        # Handle mimic command directly here - hard coded
        if text_input.lower().startswith('mimic'):
            mimic_text = text_input[5:].lstrip(': ').strip()
            logger.info(f"Mimic command detected, returning: {mimic_text[:30]}...")
            return mimic_text
        
        if not self.connected:
            await self.connect()
            if not self.connected:
                return "I'm unable to connect to my thinking module at the moment."
        
        try:
            # Format the prompt
            formatted_prompt = self.format_prompt(text_input)
            
            # Configure shorter max_length to avoid garbled responses
            max_tokens = 150  # This should be enough for 2-3 sentences
            
            # Prepare the generation parameters
            generation_data = {
                "prompt": formatted_prompt,
                "max_length": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 30,  # Reduced for more focused responses
                "repetition_penalty": 1.2,
                "stop_sequence": ["\nCurrent Query:", "\n\nCurrent Query:", "Human:", "\nUser:", "USER:"]
            }
            
            # Send the request
            response = requests.post(
                f"{self.server_url}/api/v1/generate",
                json=generation_data,
                timeout=60  # 60 seconds timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('results', [{}])[0].get('text', '')
                
                # Clean up the text
                for stop_seq in generation_data["stop_sequence"]:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                
                # Apply additional cleaning
                cleaned_text = self.clean_response(generated_text)
                
                # Check for an empty response after cleaning
                if not cleaned_text:
                    return "I acknowledge your query."
                    
                return cleaned_text
            else:
                logger.error(f"KoboldCpp server error: {response.status_code} - {response.text}")
                return "I encountered an error processing your request."
                
        except requests.exceptions.Timeout:
            logger.error("KoboldCpp request timed out")
            self.connected = False
            return "I'm taking too long to think. Please try again with a simpler question."
            
        except Exception as e:
            logger.error(f"Error communicating with KoboldCpp server: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.connected = False
            return "I encountered an error processing your request."

# Add a callback function for Gemini responses
def handle_gemini_response(response_text):
    last_responses.append(response_text)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('Shutting down processes...')
    shutdown_event.set()
    if tts_process:
        tts_process.terminate()
#    if rvc_process:
#        rvc_process.terminate()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
print_memory_usage("Memory usage before pipeline start")
async def init_llm_client():
    """Initialize the LLM client"""
    global llm_client
    
    llm_client = KoboldLLMClient(server_url="http://192.168.1.3:5001")
    connected = await llm_client.connect()
    
    if connected:
        processes_ready['gemini'].set()
        print(f"\033[92m[{time.strftime('%H:%M:%S')}] LLM client ready\033[0m")
    else:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to connect to LLM server\033[0m")
        
    # Start background task to process the queue
    asyncio.create_task(process_llm_queue())
    
async def process_llm_queue():
    """Process messages from the gemini_queue"""
    while not shutdown_event.is_set():
        try:
            # Check if there's a message in the queue
            try:
                text_input = gemini_queue.get_nowait()
                if text_input:
                    print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing LLM input: {text_input}\033[0m")
                    
                    # Get response from LLM server
                    gemini_output = await llm_client.get_response(text_input)
                    
                    if gemini_output:
                        print(f"\033[95m[{time.strftime('%H:%M:%S')}] LLM response: {gemini_output}\033[0m")
                        
                        # Add the audio filename to the log
                        audio_filename = f"out_{time.strftime('%Y%m%d_%H%M%S')}_{response_counter}.wav"
                        chat_logger.log_interaction(text_input, gemini_output, audio_filename)
                        
                        # Add to last_responses
                        last_responses.append(gemini_output)
                        
                        # Forward to TTS if ready
                        if processes_ready['tts'].is_set():
                            tts_queue.put(gemini_output)
                            print(f"\033[95m[{time.strftime('%H:%M:%S')}] Sent to TTS queue: {gemini_output}\033[0m")
            except queue.Empty:
                pass
                
            # Prevent CPU spinning
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in LLM queue processing: {str(e)}")
            await asyncio.sleep(1)  # Back off on error


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

def setup_msvc_environment():
    """Setup MSVC environment for CUDA compilation"""
    import os
    
    # Direct path to cl.exe that you provided
    cl_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
    msvc_base = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808"
    
    if os.path.exists(os.path.join(cl_path, "cl.exe")):
        print(f"\033[92m[{time.strftime('%H:%M:%S')}] Using MSVC compiler at: {cl_path}\033[0m")
        
        # Add to PATH at the beginning to ensure it's found first
        os.environ["PATH"] = cl_path + os.pathsep + os.environ["PATH"]
        
        # Set VS140COMNTOOLS environment variable (may be needed by some build systems)
        vs_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools"
        if os.path.exists(vs_path):
            os.environ["VS140COMNTOOLS"] = vs_path
        
        # Set other necessary environment variables
        os.environ["INCLUDE"] = os.path.join(msvc_base, "include")
        os.environ["LIB"] = os.path.join(msvc_base, "lib", "x64")
        
        return True
    
    print(f"\033[93m[{time.strftime('%H:%M:%S')}] MSVC compiler not found at specified path\033[0m")
    return False

def init_bigvgan():
    """Initialize BigVGAN model"""
    global bigvgan_model
    try:
        from BigVGAN import bigvgan
        
        # Find BigVGAN path
        bigvgan_model_paths = [
            os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x"),
            os.path.join(current_dir, "GPT-SoVITS", "GPT_SoVITS", "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x"),
            os.path.join(current_dir, "GPT_SoVITS", "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x")
        ]
        
        bigvgan_path = None
        for path in bigvgan_model_paths:
            if os.path.exists(path):
                bigvgan_path = path
                print(f"\033[94m[{time.strftime('%H:%M:%S')}] Found BigVGAN at: {path}\033[0m")
                break
                
        if not bigvgan_path:
            raise FileNotFoundError("BigVGAN model not found in any expected location")
            
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Loading BigVGAN from: {bigvgan_path}\033[0m")
        
        # Setup MSVC environment for CUDA compilation
        msvc_available = setup_msvc_environment()
        
        # Try with CUDA kernels if MSVC is available
        if msvc_available:
            try:
                # First attempt with CUDA kernels
                bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                    bigvgan_path,
                    local_files_only=True,
                    use_cuda_kernel=True
                )
            except Exception as cuda_err:
                print(f"\033[93m[{time.strftime('%H:%M:%S')}] Failed to load with CUDA kernels: {str(cuda_err)}. Falling back to CPU kernels.\033[0m")
                # Fallback to CPU kernels if CUDA fails
                bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                    bigvgan_path,
                    local_files_only=True,
                    use_cuda_kernel=False
                )
        else:
            # MSVC not available, use CPU kernels
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                bigvgan_path,
                local_files_only=True,
                use_cuda_kernel=False
            )
        
        # Remove weight norm and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval()
        
        # Set proper device and precision
        if is_half:
            bigvgan_model = bigvgan_model.half().to(device)
        else:
            bigvgan_model = bigvgan_model.to(device)

        first_param = next(bigvgan_model.parameters(), None)
        param_dtype = first_param.dtype if first_param is not None else "unknown"
        print(f"\033[91m[MODEL] BigVGAN model loaded with precision: {param_dtype}\033[0m")
        
        print(f"\033[92m[{time.strftime('%H:%M:%S')}] [MODEL] BigVGAN model loaded with precision: {param_dtype}, model initialized successfully\033[0m")
        return bigvgan_model
        
    except Exception as e:
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to initialize BigVGAN: {str(e)}\033[0m")
        import traceback
        traceback.print_exc()
        return None

def ensure_bigvgan():
    """Download BigVGAN model if it doesn't exist"""
    # Define the path where BigVGAN should be
    bigvgan_path = os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x")
    
    # Update SOVITS_CONFIG to use this path
    SOVITS_CONFIG["bigvgan_path"] = bigvgan_path
    
    if not os.path.exists(bigvgan_path):
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] BigVGAN model not found at {bigvgan_path}. Downloading...\033[0m")
        os.makedirs(bigvgan_path, exist_ok=True)
        
        # Download directly from huggingface
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="nvidia/bigvgan_v2_24khz_100band_256x", 
                local_dir=bigvgan_path
            )
            print(f"\033[92m[{time.strftime('%H:%M:%S')}] BigVGAN model downloaded to {bigvgan_path}\033[0m")
        except Exception as e:
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to download BigVGAN: {str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise
    else:
        print(f"\033[92m[{time.strftime('%H:%M:%S')}] BigVGAN model found at {bigvgan_path}\033[0m")

def ensure_sovitsv3():
    """Ensure SoVITS v3 model path is correctly set"""
    # Define the absolute path where v3 model should be
    sovitsv3_path = os.path.join(gpt_sovits_path, "GPT_SoVITS", "pretrained_models", "s2Gv3.pth")
    
    print(f"\033[94m[{time.strftime('%H:%M:%S')}] Looking for SoVITS v3 model at: {sovitsv3_path}\033[0m")
    
    # Update SOVITS_CONFIG
    SOVITS_CONFIG["sovitsv3_path"] = sovitsv3_path
    
    # Check if file exists
    if not os.path.exists(sovitsv3_path):
        print(f"\033[93m[{time.strftime('%H:%M:%S')}] Warning: SoVITS v3 model not found at {sovitsv3_path}\033[0m")
        print(f"\033[93m[{time.strftime('%H:%M:%S')}] V3 features will be disabled\033[0m")
    else:
        print(f"\033[92m[{time.strftime('%H:%M:%S')}] SoVITS v3 model found at {sovitsv3_path}\033[0m")
    
    # Set environment variable for other modules to use
    os.environ["sovitsv3_path"] = sovitsv3_path


def run_tts_cli():
    try:
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Initializing GPT-SoVITS processing...\033[0m")
        
        # Make sure output directories exist
        os.makedirs(os.path.join("audio", "out"), exist_ok=True)
        os.makedirs(os.path.join("audio", "temp"), exist_ok=True)
        
        # Ensure BigVGAN model is available
        ensure_bigvgan()
        ensure_sovitsv3()

        print(f"\033[97m[SYSTEM] CUDA is {'available' if torch.cuda.is_available() else 'not available'}\033[0m")
        if torch.cuda.is_available():
            print(f"\033[97m[SYSTEM] GPU: {torch.cuda.get_device_name()}\033[0m")
            print(f"\033[97m[SYSTEM] CUDA capability: {torch.cuda.get_device_capability()}\033[0m")
            print(f"\033[97m[SYSTEM] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\033[0m")
            print(f"\033[97m[SYSTEM] CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\033[0m")

        
        # Initialize models
        try:
            # Initialize TTS directly
            from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
            
            # Load model weights
            change_gpt_weights(SOVITS_CONFIG["gpt_path"])
            
            # Note: change_sovits_weights returns a generator, we need to consume it
            generator = change_sovits_weights(SOVITS_CONFIG["sovits_path"])
            try:
                next(generator)
            except StopIteration:
                pass  # This is expected
            
            # Initialize BigVGAN if we're using v3 model
            if "v3" in SOVITS_CONFIG["sovits_path"]:
                init_bigvgan()
            
            # Set flag that TTS is ready
            processes_ready['tts'].set()
            print(f"\033[92m[{time.strftime('%H:%M:%S')}] GPT-SoVITS initialization complete\033[0m")
            
        except Exception as e:
            print(f"\033[91m[{time.strftime('%H:%M:%S')}] Failed to initialize GPT-SoVITS: {e}\033[0m")
            import traceback
            traceback.print_exc()
            return

        # Main processing loop
        while not shutdown_event.is_set():
            try:
                text = tts_queue.get(timeout=1)
                if text is None:
                    break
                    
                if not text:
                    continue
                
                # Process text 
                text = text.strip()
                if not text:
                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Empty text after sanitization\033[0m")
                    continue
                    
                print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing TTS: {text}\033[0m")
                
                try:
                    # Process text in chunks to handle long responses better
                    chunks = process_text_in_chunks(text)
                    print(f"\033[95m[{time.strftime('%H:%M:%S')}] Processing text in {len(chunks)} chunks\033[0m")
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    sovits_output = os.path.abspath(os.path.join("audio", "temp", f"sovits_{timestamp}.wav"))
                    
                    # Create a direct patch to the get_tts_wav function
                    # This handles the input tensor conversion directly
                    # Replace the tts_wav_patched function in web.py with this version
                    def tts_wav_patched(*args, **kwargs):
                        """Safely wrap the get_tts_wav generator to handle StopIteration gracefully"""
                        # Get the original generator
                        result = get_tts_wav(*args, **kwargs)
                        
                        # Store intermediate results
                        final_result = None
                        
                        def safe_wrapper():
                            nonlocal final_result
                            try:
                                for item in result:
                                    final_result = item  # Save the most recent result
                                    yield item
                            except Exception as e:
                                if isinstance(e, StopIteration):
                                    # This is normal generator behavior, not an error
                                    if final_result:
                                        # If we have a final result, return it
                                        print(f"\033[92m[{time.strftime('%H:%M:%S')}] Generator completed with final result\033[0m")
                                        yield final_result
                                    else:
                                        # If generator ended without producing a result, this is a real error
                                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Generator ended without producing a result\033[0m")
                                        raise RuntimeError("TTS generator failed to produce any output")
                                else:
                                    # For other errors, log and re-raise
                                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] TTS Error: {str(e)}\033[0m")
                                    if "Input type" in str(e) and "weight type" in str(e):
                                        print(f"\033[91m[ERROR] Tensor type mismatch. Please manually set all tensors to the same precision.\033[0m")
                                    raise e
                        
                        return safe_wrapper()

                    # Use the function as before:
                    synthesis_result = tts_wav_patched(
                        ref_wav_path=SOVITS_CONFIG["ref_audio"],
                        prompt_text=SOVITS_CONFIG["ref_text"],
                        prompt_language=SOVITS_CONFIG["ref_language"],
                        text=text,
                        text_language=SOVITS_CONFIG["output_language"],
                        how_to_cut=SOVITS_CONFIG.get("how_to_cut", i18n("凑四句一切")),
                        top_k=int(SOVITS_CONFIG["tuning_params"]["top_k"]),
                        top_p=float(SOVITS_CONFIG["tuning_params"]["top_p"]),
                        temperature=float(SOVITS_CONFIG["tuning_params"]["temperature"]),
                        speed=float(SOVITS_CONFIG["tuning_params"]["speed_factor"]),
                        sample_steps=8  # Default for v3 models
                    )

                    # Use a safer approach to extract results from the generator
                    result_list = []
                    try:
                        # Process the generator one item at a time
                        for item in synthesis_result:
                            if item is not None:
                                result_list.append(item)
                            else:
                                print(f"\033[93m[{time.strftime('%H:%M:%S')}] Warning: Received None result from TTS\033[0m")
                    except Exception as e:
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Error processing TTS results: {str(e)}\033[0m")
                        import traceback
                        print(f"\033[91m[{time.strftime('%H:%M:%S')}] {traceback.format_exc()}\033[0m")

                    # Process the results if we have any
                    if result_list:
                        last_result = result_list[-1]  # Take the last result
                        sample_rate, audio_data = last_result
                        
                        # Save the output
                        import soundfile as sf
                        print_memory_usage("On TTS output write")
                        sf.write(sovits_output, audio_data, sample_rate)
                        
                        print(f"\033[92m[{time.strftime('%H:%M:%S')}] TTS output saved to {sovits_output}\033[0m")
                        
                        # Process with RVC
                        rvc_queue.put(sovits_output)
                        final_output = os.path.join("audio", "out", f"out_{timestamp}.wav")
                        
                        # Wait for RVC to complete (with timeout)
                        start_time = time.time()
                        timeout_seconds = 90  # More generous timeout
                        rvc_completed = False

                        # Correctly construct the expected file paths
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        final_output = os.path.join("audio", "out", f"out_{timestamp}.wav")
                        latest_output = os.path.join("audio", "out", "out.wav")

                        # Check both the timestamped output and any RVC output
                        while time.time() - start_time < timeout_seconds:
                            # Check for the timestamped output
                            if os.path.exists(final_output) and os.path.getsize(final_output) > 1000:
                                print(f"\033[92m[{time.strftime('%H:%M:%S')}] RVC output found: {final_output}\033[0m")
                                print_memory_usage("After RVC output")
                                rvc_completed = True
                                break
                            
                            # Also check for any newer RVC outputs that might have a different timestamp
                            rvc_outputs = [f for f in os.listdir(os.path.join("audio", "out")) 
                                        if f.startswith("out_") and f.endswith(".wav")]
                            if rvc_outputs:
                                newest_output = max(rvc_outputs, key=lambda x: os.path.getmtime(os.path.join("audio", "out", x)))
                                newest_path = os.path.join("audio", "out", newest_output)
                                
                                # Check if this is a new file (created after we started waiting)
                                if os.path.getmtime(newest_path) > start_time and os.path.getsize(newest_path) > 1000:
                                    print(f"\033[92m[{time.strftime('%H:%M:%S')}] Found newer RVC output: {newest_path}\033[0m")
                                    final_output = newest_path
                                    rvc_completed = True
                                    break
                            
                            time.sleep(0.5)

                        # After timeout or finding the file
                        if not rvc_completed:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC processing timeout after {timeout_seconds} seconds\033[0m")
                            
                            # Do one final check for any RVC outputs before falling back
                            rvc_outputs = [f for f in os.listdir(os.path.join("audio", "out")) 
                                        if f.startswith("out_") and f.endswith(".wav")]
                            if rvc_outputs:
                                newest_output = max(rvc_outputs, key=lambda x: os.path.getmtime(os.path.join("audio", "out", x)))
                                newest_path = os.path.join("audio", "out", newest_output)
                                
                                # Check if file size is valid
                                if os.path.getsize(newest_path) > 1000:
                                    print(f"\033[92m[{time.strftime('%H:%M:%S')}] Found valid RVC output after timeout: {newest_path}\033[0m")
                                    final_output = newest_path
                                    rvc_completed = True
                            
                            # Only use fallback if no valid RVC output was found
                            if not rvc_completed:
                                if os.path.exists(sovits_output) and os.path.getsize(sovits_output) > 1000:
                                    print(f"\033[93m[{time.strftime('%H:%M:%S')}] Using SoVITS output directly as fallback\033[0m")
                                    shutil.copy2(sovits_output, final_output)
                                    rvc_completed = True
                                else:
                                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] No valid TTS results produced\033[0m")

                        # Update the latest output file
                        if rvc_completed:
                            # Remove existing file if it exists
                            if os.path.exists(latest_output):
                                try:
                                    os.remove(latest_output)
                                except Exception as e:
                                    print(f"\033[91m[{time.strftime('%H:%M:%S')}] Error cleaning old audio: {e}\033[0m")
                            
                            # Copy the final output to out.wav
                            try:
                                shutil.copy2(final_output, latest_output)
                                print(f"\033[92m[{time.strftime('%H:%M:%S')}] Final output copied to {latest_output}\033[0m")
                            except Exception as e:
                                print(f"\033[91m[{time.strftime('%H:%M:%S')}] Error copying final output: {e}\033[0m")

                        # Clean up temp file
                        try:
                            if os.path.exists(sovits_output):
                                os.remove(sovits_output)
                        except Exception as e:
                            print(f"\033[93m[{time.strftime('%H:%M:%S')}] Error cleaning temp file: {e}\033[0m")
                            
                        if rvc_completed:
                            print(f"\033[92m[{time.strftime('%H:%M:%S')}] Setting TTS completion flag\033[0m")
                            tts_completed.set()
                        else:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] No valid output produced\033[0m")
                        
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
                        output_received = False

                        while not output_received and time.time() - start_time < RVC_TIMEOUT:
                            line = rvc_process.stdout.readline().strip()
                            if line:
                                output = line
                                output_received = True
                                print(f"\033[94m[{time.strftime('%H:%M:%S')}] RVC output: {output}\033[0m")
                                
                                # Wait a short time for file to be fully written
                                time.sleep(1)
                                
                                # Explicitly check if the output file exists before continuing
                                if 'Converting audio' in output:
                                    # Parse the output path from the message
                                    match = re.search(r'audio\\out\\(out_[^\.]+\.wav)', output)
                                    if match:
                                        output_file = match.group(1)
                                        output_path = os.path.join("audio", "out", output_file)
                                        if os.path.exists(output_path):
                                            print(f"\033[92m[{time.strftime('%H:%M:%S')}] Confirmed RVC output exists: {output_path}\033[0m")
                                            # Set completion flag here directly
                                            tts_completed.set()
                                            break
                                
                            # Check for timeout
                            if time.time() - start_time >= RVC_TIMEOUT:
                                break
                                
                            time.sleep(0.1)

                        if not output_received:
                            print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC response timeout\033[0m")
                        
                        print(f"\033[94m[{time.strftime('%H:%M:%S')}] RVC output: {output}\033[0m")

            except queue.Empty:
                continue
            except TimeoutError as te:
                print(f"\033[91m[{time.strftime('%H:%M:%S')}] RVC Timeout: {str(te)}\033[0m")
                # Reset RVC process
                if rvc_process:
                    try:
                        rvc_process.terminate()
                    except:
                        pass
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

@app.route('/get_response', methods=['GET'])
def get_response():
    audio_ready = tts_completed.is_set()
    latest_output = os.path.join("audio", "out", "out.wav")
    
    # Double check if the audio file actually exists
    if audio_ready and not os.path.exists(latest_output):
        print(f"\033[91m[{time.strftime('%H:%M:%S')}] Audio reported ready but file is missing\033[0m")
        audio_ready = False
        tts_completed.clear()
    
    if last_responses:
        return jsonify({
            'response': last_responses[-1],
            'audio_ready': audio_ready
        })
    return jsonify({
        'response': None,
        'audio_ready': False
    })

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
            
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] Starting RVC process...\033[0m")
        rvc_thread = threading.Thread(target=run_rvc_cli, daemon=True)
        rvc_thread.start()
        threads.append(rvc_thread)
        
        # We'll initialize the LLM client in the async part
        print(f"\033[94m[{time.strftime('%H:%M:%S')}] LLM client will be initialized in async context\033[0m")
        
        start_time = time.time()
        # We only wait for TTS and RVC here, LLM client will be initialized later
        while not (processes_ready['tts'].is_set() and processes_ready['rvc'].is_set()):
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
    
    # Disconnect LLM client
    if llm_client:
        await llm_client.disconnect()

async def run_all():
    # Initialize all the background threads first
    threads = init_threads()
    test_queues()
    time.sleep(2)
    check_process_status()
    
    # Initialize LLM client
    await init_llm_client()
    
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
        
        async def startup():
            # Initialize LLM client
            await init_llm_client()
            
            # Run both servers
            await asyncio.gather(
                run_flask(),
                run_discord()
            )
            
        try:
            # Run both servers
            logger.info("Starting servers...")
            loop.run_until_complete(startup())
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
            
            # Disconnect LLM client
            if llm_client:
                loop.run_until_complete(llm_client.disconnect())
            
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