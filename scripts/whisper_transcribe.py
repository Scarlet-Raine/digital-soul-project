import os
from pathlib import Path
from faster_whisper import WhisperModel
import torch

def transcribe_dataset(data_dir, model_size="large-v3"):
    """
    Transcribe audio files using Faster Whisper
    """
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Loading Whisper model {model_size} on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    data_path = Path(data_dir)
    wavs_dir = data_path / "wavs"
    
    # Read existing filelists
    def read_filelist(filename):
        filepath = data_path / "filelists" / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
            
    def write_filelist(filename, lines):
        filepath = data_path / "filelists" / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(f"{line}\n")

    # Process train and val lists
    for list_file in ["train.list", "val.list"]:
        print(f"\nProcessing {list_file}...")
        lines = read_filelist(list_file)
        new_lines = []
        
        for i, line in enumerate(lines):
            wav_path = line.split('|')[0]
            full_path = str(data_path / wav_path)
            
            print(f"Transcribing {i+1}/{len(lines)}: {wav_path}")
            
            # Run transcription
            segments, info = model.transcribe(
                full_path,
                language="ja",
                task="transcribe",
                beam_size=5,
                vad_filter=True
            )
            
            # Get the transcription
            text = ' '.join([segment.text for segment in segments]).strip()
            
            # Update line with transcription
            new_line = f"{wav_path}|{text}|"
            new_lines.append(new_line)
            
            # Print progress
            print(f"Text: {text}")
            
        # Save updated filelist
        write_filelist(list_file, new_lines)
        print(f"Updated {list_file}")

if __name__ == "__main__":
    data_dir = r"C:\Users\EVO\Documents\AI\GPT-SoVits\data\2b"
    transcribe_dataset(data_dir)