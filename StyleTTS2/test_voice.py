import os
import time
import yaml
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
import sys
import random
from tts_cli import compute_style, inference, LFinference, process_text, initialize_models, analyze_pitch, compute_pitch_shift

class VoiceTester:
    def __init__(self):
        print("Initializing models...")
        initialize_models()
        self.output_dir = "voice_tests"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Use multiple voice samples if available
        self.voice_sources = [
            './voice/voice_2b_short.wav',
            # Add paths to other Japanese voice clips
        ]
        
        print("Computing reference style...")
        with torch.no_grad():
            styles = []
            for source in self.voice_sources:
                if os.path.exists(source):
                    styles.append(compute_style(source))
            self.ref_s = torch.mean(torch.stack(styles), dim=0) if styles else compute_style(self.voice_sources[0])
        
        # Compute base pitch adjustment
        temp_output = os.path.join(self.output_dir, "temp_output.wav")
        self.base_pitch_shift = compute_pitch_shift(self.voice_sources[0], temp_output) if os.path.exists(temp_output) else 0
        print(f"Base pitch shift computed: {self.base_pitch_shift:.2f} semitones")
        
        self.test_phrase = (
            "Commander, listen carefully. As YoRHa Type B number 2, "
            "I am equipped with advanced combat capabilities. "
            "My primary objective is battlefield superiority, "
            "and I will eliminate any machine threats we encounter. "
            "You can trust in my abilities to complete this mission."
        )

    def save_audio(self, wav_data, filename):
        filepath = os.path.join(self.output_dir, filename)
        write(filepath, 24000, wav_data)
        print(f"Saved: {filepath}")

    def test_fine_tuning(self):
        settings = [
            {
                'name': 'determined_base',
                'alpha': 0.5,
                'beta': 0.8,
                't': 0.8,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.2
            },
            {
                'name': 'determined_hybrid',
                'alpha': 0.6,
                'beta': 0.85,
                't': 0.85,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.1
            }
    ]

    print("\nTesting fine-tuned settings...")
    s_prev = None
    for setting in settings:
        try:
            # Remove language parameter from process_text call
            processed_phrase = process_text(self.test_phrase)
            if processed_phrase is None:
                continue
                
            print(f"\nTesting {setting['name']}...")
            start = time.time()
            
            with torch.no_grad():
                wav, s_prev = LFinference(
                    processed_phrase, 
                    s_prev, 
                    self.ref_s,
                    alpha=setting['alpha'],
                    beta=setting['beta'],
                    t=setting['t'],
                    diffusion_steps=35,
                    embedding_scale=2.5,
                    pitch_adjust=setting['pitch_adjust'],
                    duration_scale=setting['duration_scale']
                )
            
            rtf = (time.time() - start) / (len(wav) / 24000)
            print(f"RTF = {rtf:5f}")
            
            filename = f"{setting['name']}_tuned.wav"
            self.save_audio(wav, filename)
            
        except Exception as e:
            print(f"Error with {setting['name']}: {str(e)}")
            import traceback
            print(traceback.format_exc())

        print("\nTesting regular inference settings...")
        for setting in settings:
            try:
                processed_phrase = process_text(self.test_phrase)
                if processed_phrase is None:
                    continue
                    
                print(f"\nTesting {setting['name']} settings...")
                start = time.time()
                
                with torch.no_grad():
                    wav = inference(
                        processed_phrase, 
                        self.ref_s,
                        diffusion_steps=setting['steps'],
                        embedding_scale=setting['scale'],
                        pitch_adjust=setting['pitch_adjust'],
                        duration_scale=setting['duration_scale']
                    )
                
                rtf = (time.time() - start) / (len(wav) / 24000)
                print(f"RTF = {rtf:5f}")
                
                filename = f"{setting['name']}_tuned.wav"
                self.save_audio(wav, filename)
                
            except Exception as e:
                print(f"Error with {setting['name']}: {str(e)}")

    def test_style_mixing(self):
        settings = [
            {
                'name': 'composed',
                'alpha': 0.3,
                'beta': 0.6,
                't': 0.6,
                'pitch_adjust': 0,
                'duration_scale': 1.0
            },
            {
                'name': 'assertive',
                'alpha': 0.4,
                'beta': 0.7,
                't': 0.7,
                'pitch_adjust': 1,
                'duration_scale': 1.1
            },
            {
                'name': 'determined',
                'alpha': 0.5,
                'beta': 0.8,
                't': 0.8,
                'pitch_adjust': 2,
                'duration_scale': 1.2
            },
            {
                'name': 'intense',
                'alpha': 0.6,
                'beta': 0.9,
                't': 0.9,
                'pitch_adjust': 3,
                'duration_scale': 1.3
            }
        ]

        print("\nTesting style-mixing inference settings...")
        s_prev = None
        for setting in settings:
            try:
                processed_phrase = process_text(self.test_phrase)
                if processed_phrase is None:
                    continue
                    
                print(f"\nTesting {setting['name']}...")
                start = time.time()
                
                with torch.no_grad():
                    wav, s_prev = LFinference(
                        processed_phrase, 
                        s_prev, 
                        self.ref_s,
                        alpha=setting['alpha'],
                        beta=setting['beta'],
                        t=setting['t'],
                        diffusion_steps=35,
                        embedding_scale=2.5
                    )
                
                rtf = (time.time() - start) / (len(wav) / 24000)
                print(f"RTF = {rtf:5f}")
                
                filename = f"{setting['name']}_tuned.wav"
                self.save_audio(wav, filename)
                
            except Exception as e:
                print(f"Error with {setting['name']}: {str(e)}")

    def run_all_tests(self):
        # Run all test sets
        self.test_inference_settings()
        self.test_style_mixing()
        
        # Print summary
        print("\nAll tests complete! Check the 'voice_tests' directory for outputs.")
        print("\nFile naming convention and settings:")
        print("\nRegular inference files:")
        print("- natural_tuned.wav: Baseline natural speech")
        print("- smooth_tuned.wav: Slower, smoother delivery")
        print("- emphatic_tuned.wav: Clear emphasis on key words")
        print("- dramatic_tuned.wav: Strong emotional delivery")
        print("\nStyle mixing files:")
        print("- composed_tuned.wav: Calm, measured delivery")
        print("- assertive_tuned.wav: Confident, direct delivery")
        print("- determined_tuned.wav: Strong, purposeful delivery")
        print("- intense_tuned.wav: Maximum emotional impact")

if __name__ == "__main__":
    print("Starting voice parameter testing...")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    tester = VoiceTester()
    tester.run_all_tests()