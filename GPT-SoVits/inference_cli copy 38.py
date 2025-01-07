import os
import sys
import numpy as np
import threading
import queue
import sounddevice as sd
import warnings
import requests
import json
from time import time
from pathlib import Path
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# Suppress torch stft warning
warnings.filterwarnings("ignore", category=UserWarning, message="stft with return_complex=False is deprecated.")

# Ensure that GPT_SoVITS is in the Python path
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, 'GPT_SoVITS'))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Default configuration
DEFAULT_CONFIG = {
    "gpt_path": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/VestiaZeta_GPT (KitLemonfoot).ckpt",
    "sovits_path": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/VestiaZeta_SoVITS (KitLemonfoot).pth",
    "ref_audio": "C:/Users/SUBSECT/Downloads/MyWaifu/dataset/wavs/19.wav",
    "ref_text": "Here's another fun fact for you. The fear of phobia. Isn't that ironic?",
    "output_dir": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/Reference Audios",
    "ollama_model": "llama3.1:8b",
    "system_prompt": "You are Hikari-Chan, an enthusiastic and quirky Waifu with a passion for technology and creativity! dont use capital letters ever",
    "n_ctx": 2048
}

def load_or_create_config(config_path="hikari_config.json"):
    """Load existing config or create new one with defaults."""
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Update with any missing default keys
                config = DEFAULT_CONFIG.copy()
                config.update(loaded_config)
                print("loaded existing configuration from", config_path)
                return config
        except Exception as e:
            print(f"error loading config ({e}), using defaults...")
            return DEFAULT_CONFIG
    else:
        # Create new config file with defaults
        save_config(DEFAULT_CONFIG, config_path)
        print("created new configuration file at", config_path)
        return DEFAULT_CONFIG

def save_config(config, config_path="hikari_config.json"):
    """Save current configuration to file."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("saved configuration to", config_path)
    except Exception as e:
        print(f"error saving config: {e}")

def get_device_id(device_name):
    """Get the device ID from a device name string."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['name'].strip() == device_name.strip():
            return i
    # If device not found, use default output device
    print(f"\nWarning: Device '{device_name}' not found, using default output device")
    return sd.default.device[1]  # Return default output device

def audio_playback_thread(audio_queue):
    """Audio playback thread that plays audio fragments from the queue."""
    stream = None
    try:
        while True:
            # Get the next audio fragment
            audio_fragment, sample_rate = audio_queue.get()
            try:
                if audio_fragment is None:
                    break

                # Ensure the audio data is in float32 format
                if audio_fragment.dtype == np.int16:
                    audio_fragment = audio_fragment.astype(np.float32) / 32768.0
                elif audio_fragment.dtype != np.float32:
                    raise ValueError(f"Unsupported audio dtype: {audio_fragment.dtype}")

                # Initialize or update the audio stream
                if stream is None or stream.samplerate != sample_rate:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    sd.default.samplerate = sample_rate
                    sd.default.channels = 1
                    stream = sd.OutputStream(
                        dtype='float32',
                        channels=1,
                        samplerate=sample_rate
                    )
                    stream.start()

                # Write the audio fragment to the stream
                stream.write(audio_fragment)
            finally:
                audio_queue.task_done()
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

def print_settings_menu():
    """Display the settings menu options."""
    print("\n=== Hikari Settings ===")
    print("1. Save Configuration")
    print("2. Back to Chat")
    print("===================")

def settings_menu():
    """Handle the settings menu interaction."""
    while True:
        print_settings_menu()
        choice = input("Enter option (1-2): ").strip()
        
        if choice == '1':
            save_config(CONFIG)
        elif choice == '2':
            return
        else:
            print("Invalid option. Please try again.")

def print_help():
    """Display available commands."""
    print("\n=== Available Commands ===")
    print("/settings - Open settings menu")
    print("/help     - Show this help")
    print("/quit     - Exit the program")
    print("=======================")

class OllamaStreamWorker:
    def __init__(self, messages, model=None):
        self.messages = messages
        self.model = model or CONFIG["ollama_model"]

    def generate(self):
        api_url = "http://localhost:11434/api/chat"
        data = {
            "model": self.model,
            "messages": self.messages,
            "stream": True
        }
        response = requests.post(api_url, json=data, stream=True)
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8'))
                chunk = json_response.get('message', {}).get('content', '')
                if chunk:
                    yield chunk

def main():
    print("\nHikari-Chan initialized! Type /help for available commands.")
    
    # Initialize conversation history with system message
    conversation_history = [
        {"role": "system", "content": CONFIG["system_prompt"]}
    ]
    
    # Create a queue for audio fragments
    audio_queue = queue.Queue(maxsize=100)

    # Start the audio playback thread
    playback_thread = threading.Thread(
        target=audio_playback_thread,
        args=(audio_queue,)
    )
    playback_thread.start()
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("Exiting the application. Goodbye!")
                break
            elif user_input.lower() == "/settings":
                settings_menu()
                continue
            elif user_input.lower() == "/help":
                print_help()
                continue
            elif not user_input:
                continue
            
            # Append user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Initialize buffers and counters
            buffer = ""
            char_count = 0
            waiting_for_punctuation = False
            assistant_buffer = ""

            # Generate and process the chat completion
            print("Assistant: ", end="", flush=True)
            ollama_worker = OllamaStreamWorker(conversation_history)
            for token in ollama_worker.generate():
                print(token, end="", flush=True)
                buffer += token
                assistant_buffer += token
                char_count += len(token)

                if not waiting_for_punctuation:
                    if char_count >= 100:
                        waiting_for_punctuation = True
                elif any(punct in token for punct in ['.', '!', '?']):
                    # Process completed sentence with TTS
                    synthesis_result = get_tts_wav(
                        ref_wav_path=CONFIG["ref_audio"],
                        prompt_text=CONFIG["ref_text"],
                        prompt_language="English",
                        text=buffer,
                        text_language="English"
                    )
                    
                    # Queue the audio fragments
                    for sampling_rate, audio_fragment in synthesis_result:
                        audio_queue.put((audio_fragment, sampling_rate))
                    
                    # Add silence between sentences
                    silence_duration = 0.5
                    silence = np.zeros(int(sampling_rate * silence_duration), dtype='float32')
                    audio_queue.put((silence, sampling_rate))
                    
                    # Reset buffers and counters
                    buffer = ""
                    char_count = 0
                    waiting_for_punctuation = False

            # Append assistant message to conversation history
            conversation_history.append({"role": "assistant", "content": assistant_buffer})

            # Process any remaining text
            if buffer.strip():
                synthesis_result = get_tts_wav(
                    ref_wav_path=CONFIG["ref_audio"],
                    prompt_text=CONFIG["ref_text"],
                    prompt_language="English",
                    text=buffer,
                    text_language="English"
                )
                
                for sampling_rate, audio_fragment in synthesis_result:
                    audio_queue.put((audio_fragment, sampling_rate))
                
                silence_duration = 0.5
                silence = np.zeros(int(sampling_rate * silence_duration), dtype='float32')
                audio_queue.put((silence, sampling_rate))

    finally:
        # Clean up
        audio_queue.put((None, None))
        audio_queue.join()
        playback_thread.join()

# Load configuration and initialize models
CONFIG = load_or_create_config()
change_gpt_weights(CONFIG["gpt_path"])
change_sovits_weights(CONFIG["sovits_path"])

if __name__ == '__main__':
    main()