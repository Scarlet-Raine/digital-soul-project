import argparse
import os
import soundfile as sf
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, next, get_tts_wav

i18n = I18nAuto()

def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, 
               target_text_path, target_language, output_path, sample_steps=8, if_sr=False):
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()
    
    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()
    
    # Change model weights
    # Note: change_sovits_weights is now a generator, so we need to consume it with next()
    change_gpt_weights(gpt_path=GPT_model_path)
    
    # Use next() to consume the generator and get the version info
    try:
        next(change_sovits_weights(sovits_path=SoVITS_model_path))
    except StopIteration:
        pass
    
    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        top_p=1,
        temperature=1,
        # Pass additional parameters needed for v3 models
        sample_steps=sample_steps,
        if_sr=if_sr
    )
   
    result_list = list(synthesis_result)
    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--gpt_model', required=True, help="Path to the GPT model file")
    parser.add_argument('--sovits_model', required=True, help="Path to the SoVITS model file")
    parser.add_argument('--ref_audio', required=True, help="Path to the reference audio file")
    parser.add_argument('--ref_text', required=True, help="Path to the reference text file")
    parser.add_argument('--ref_language', required=True, 
                        choices=["中文", "英文", "日文", "粤语", "韩文"], 
                        help="Language of the reference audio")
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--target_language', required=True, 
                        choices=["中文", "英文", "日文", "粤语", "韩文", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], 
                        help="Language of the target text")
    parser.add_argument('--output_path', required=True, help="Path to the output directory")
    
    # Add v3-specific parameters
    parser.add_argument('--sample_steps', type=int, default=8, 
                       choices=[4, 8, 16, 32],
                       help="Sample steps for v3 models (4, 8, 16, or 32)")
    parser.add_argument('--use_sr', action='store_true', 
                       help="Use super-resolution for v3 models (may help with muffled output)")

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
        args.sample_steps,
        args.use_sr
    )

if __name__ == '__main__':
    main()