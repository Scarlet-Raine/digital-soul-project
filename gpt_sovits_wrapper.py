
import os
import sys
import json
import time

# Get the GPT-SoVITS root directory
gpt_sovits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT-SoVITS")
gpt_sovits_module_dir = os.path.join(gpt_sovits_dir, "GPT_SoVITS")

# Add all necessary directories to Python path
sys.path.insert(0, gpt_sovits_dir)
sys.path.insert(0, gpt_sovits_module_dir)
sys.path.insert(0, os.path.join(gpt_sovits_dir, "tools"))

# Import necessary modules manually with proper path handling
import importlib.util
import importlib

# Set up environment variables
os.environ["version"] = "v3"

# Create a simplified API wrapper for inference_webui
def main():
    try:
        # Import TTS modules
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
        # Initialize TTS config
        config_path = os.path.join(gpt_sovits_module_dir, "configs", "tts_infer.yaml")
        tts_config = TTS_Config(config_path)
        tts_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_config.is_half = tts_config.device == "cuda"
        tts_config.version = "v3"
        
        # Use environment variables for paths
        tts_config.t2s_weights_path = os.environ.get("gpt_path")
        tts_config.vits_weights_path = os.environ.get("sovits_path")
        tts_config.cnhuhbert_base_path = os.environ.get("cnhubert_base_path")
        tts_config.bert_base_path = os.environ.get("bert_path")
        
        print(f"Initializing TTS with config: {tts_config}")
        tts_pipeline = TTS(tts_config)
        
        # Signal that initialization is complete
        print("GPT-SoVITS ready")
        
        # Handle requests
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue
                
            try:
                # Parse JSON request
                request = json.loads(line)
                
                # Extract parameters
                ref_audio_path = request.get("ref_audio_path")
                prompt_text = request.get("prompt_text", "")
                prompt_language = request.get("prompt_language", "中文")
                text = request.get("text", "")
                text_language = request.get("text_language", "中文")
                how_to_cut = request.get("how_to_cut", "凑四句一切")
                top_k = request.get("top_k", 20)
                top_p = request.get("top_p", 0.7)
                temperature = request.get("temperature", 0.7)
                speed = request.get("speed", 1.0)
                ref_free = request.get("ref_free", False)
                if_sr = request.get("if_sr", False)
                sample_steps = request.get("sample_steps", 32)
                output_path = request.get("output_path", "output.wav")
                
                # Import dictionaries for language and method
                from GPT_SoVITS.inference_webui import dict_language, cut_method
                
                # Create inputs dictionary for TTS
                inputs = {
                    "text": text,
                    "text_lang": dict_language.get(text_language, "en"),
                    "ref_audio_path": ref_audio_path,
                    "aux_ref_audio_paths": [],
                    "prompt_text": prompt_text,
                    "prompt_lang": dict_language.get(prompt_language, "en"),
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "text_split_method": cut_method.get(how_to_cut, "cut1"),
                    "batch_size": 1,
                    "speed_factor": speed,
                    "split_bucket": False,
                    "return_fragment": False,
                    "fragment_interval": 0.3,
                    "seed": 1234,
                    "parallel_infer": False,
                    "repetition_penalty": 1.0
                }
                
                # Generate audio
                import soundfile as sf
                for sr, audio in tts_pipeline.run(inputs):
                    # Save the audio
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    sf.write(output_path, audio, sr)
                    print(json.dumps({"success": True, "output_path": output_path}))
                    break
                    
            except Exception as e:
                error_msg = str(e)
                print(json.dumps({"success": False, "error": error_msg}))
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Import torch here to ensure it's available
    import torch
    main()
