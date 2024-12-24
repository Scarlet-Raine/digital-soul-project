import os
import time
import yaml
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
import sys
import random
from tts_cli import compute_style, inference, LFinference, process_text, initialize_models, compute_pitch_shift

class VoiceTester:
    def __init__(self):
        print("Initializing models...")
        initialize_models()
        self.output_dir = "voice_tests"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.voice_source = './voice/voice_2b_short.wav'
        print("Computing reference style...")
        with torch.no_grad():
            self.ref_s = compute_style(self.voice_source)
        
        # Compute base pitch adjustment
        temp_output = os.path.join(self.output_dir, "temp_output.wav")
        self.base_pitch_shift = compute_pitch_shift(self.voice_source, temp_output) if os.path.exists(temp_output) else 0
        print(f"Base pitch shift computed: {self.base_pitch_shift:.2f} semitones")
        
        self.test_phrase = (
            "Commander, listen carefully. As YoRHa Type B number 2, "
            "I am equipped with advanced combat capabilities. "
            "My primary objective is battlefield superiority, "
            "and I will eliminate any machine threats we encounter. "
            "You can trust in my abilities to complete this mission."
        )

    def save_audio(self, wav_data, filename):
        """Save audio data to a file"""
        filepath = os.path.join(self.output_dir, filename)
        write(filepath, 24000, wav_data)
        print(f"Saved: {filepath}")

    def test_fine_tuning(self):
        settings = [
            {
                'name': 'determined_base',  # Our reference point
                'alpha': 0.5,
                'beta': 0.8,
                't': 0.8,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.2
            },
            {
                'name': 'determined_flow1',  # More emphasis on pauses
                'alpha': 0.5,
                'beta': 0.8,
                't': 0.8,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.3  # Slightly longer pauses
            },
            {
                'name': 'determined_flow2',  # Smoother transitions
                'alpha': 0.45,
                'beta': 0.85,
                't': 0.85,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.25
            },
            {
                'name': 'determined_flow3',  # Enhanced prosody
                'alpha': 0.5,
                'beta': 0.85,
                't': 0.8,
                'pitch_adjust': self.base_pitch_shift,  # Slightly more pitch variation
                'duration_scale': 1.2
            },
            {
                'name': 'determined_flow4',  # Focus on style consistency
                'alpha': 0.45,
                'beta': 0.9,
                't': 0.9,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.2
            },
            {
                'name': 'determined_hybrid',  # Blend of best elements
                'alpha': 0.45,
                'beta': 0.85,
                't': 0.85,
                'pitch_adjust': self.base_pitch_shift,
                'duration_scale': 1.25
            }
        ]

        print("\nTesting fine-tuned settings...")
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
                
                filename = f"{setting['name']}_fine.wav"
                self.save_audio(wav, filename)
                
            except Exception as e:
                print(f"Error with {setting['name']}: {str(e)}")
                import traceback
                print(traceback.format_exc())

    def run_fine_tuning(self):
        print("\nStarting fine-tuning tests...")
        self.test_fine_tuning()
        
        print("\nFine-tuning complete! Check the 'voice_tests' directory for outputs.")
        print("\nFile variations:")
        print("- determined_base_fine.wav: Original best settings")
        print("- determined_flow1_fine.wav: Enhanced pausing")
        print("- determined_flow2_fine.wav: Smoother transitions")
        print("- determined_flow3_fine.wav: Enhanced prosody")
        print("- determined_flow4_fine.wav: Better style consistency")
        print("- determined_hybrid_fine.wav: Balanced optimization")

if __name__ == "__main__":
    print("Starting voice fine-tuning...")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    tester = VoiceTester()
    tester.run_fine_tuning()