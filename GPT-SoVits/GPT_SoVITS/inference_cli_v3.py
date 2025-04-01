import argparse
import os
import soundfile as sf
import json
from tools.i18n.i18n import I18nAuto, scan_language_list

# Import directly from inference_webui for v3 support
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, 
               target_text_path, target_language, output_path, top_k=20, top_p=0.6, 
               temperature=0.6, speed=1.0, sample_steps=32, if_sr=False):
    
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()
    
    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()
    
    # Change model weights
    print(f"Loading GPT model from: {GPT_model_path}")
    change_gpt_weights(GPT_model_path)
    
    # The next function returns a generator in v3, we need to handle it differently
    print(f"Loading SoVITS model from: {SoVITS_model_path}")
    for _ in change_sovits_weights(SoVITS_model_path):
        # Just iterate the generator to complete the loading
        pass
    
    print(f"Starting synthesis with parameters:")
    print(f"Reference audio: {ref_audio_path}")
    print(f"Reference text: {ref_text[:50]}...")
    print(f"Reference language: {ref_language}")
    print(f"Target text: {target_text[:50]}...")
    print(f"Target language: {target_language}")
    print(f"Model parameters: top_k={top_k}, top_p={top_p}, temperature={temperature}, speed={speed}")
    print(f"V3 specific: sample_steps={sample_steps}, super-resolution={if_sr}")
    
    # Synthesize audio with v3 support
    generator = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        speed=speed,
        sample_steps=sample_steps,
        if_sr=if_sr
    )
    
    # Process the generator results
    for result in generator:
        sampling_rate, audio_data = result
        
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the final audio
    output_wav_path = os.path.join(output_path, "output.wav")
    sf.write(output_wav_path, audio_data, sampling_rate)
    print(f"Audio saved to {output_wav_path}")
    
    return output_wav_path

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool (V3 Compatible)")
    
    # Basic parameters
    parser.add_argument('--gpt_model', required=True, help="Path to the GPT model file")
    parser.add_argument('--sovits_model', required=True, help="Path to the SoVITS model file")
    parser.add_argument('--ref_audio', required=True, help="Path to the reference audio file")
    parser.add_argument('--ref_text', required=True, help="Path to the reference text file")
    
    # Language support for v3
    language_choices = ["中文", "英文", "日文", "粤语", "韩文", "中英混合", "日英混合", 
                         "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"]
    parser.add_argument('--ref_language', required=True, choices=language_choices, 
                        help="Language of the reference audio")
    parser.add_argument('--target_language', required=True, choices=language_choices, 
                        help="Language of the target text")
    
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--output_path', required=True, help="Path to the output directory")
    
    # Additional tuning parameters
    parser.add_argument('--top_k', type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument('--top_p', type=float, default=0.6, help="Top-p sampling parameter")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument('--speed', type=float, default=1.0, help="Speed factor for speech")
    
    # V3 specific parameters
    parser.add_argument('--sample_steps', type=int, default=32, 
                        choices=[4, 8, 16, 32], help="Sample steps for v3 models")
    parser.add_argument('--if_sr', action='store_true', 
                        help="Enable super-resolution for v3 models")
    
    args = parser.parse_args()
    
    synthesize(
        args.gpt_model, 
        args.sovits_model, 
        args.ref_audio, 
        args.ref_text, 
        args.ref_language, 
        args.target_text, 
        args.target_language, 
        args.output_path,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
        sample_steps=args.sample_steps,
        if_sr=args.if_sr
    )

if __name__ == '__main__':
    main()