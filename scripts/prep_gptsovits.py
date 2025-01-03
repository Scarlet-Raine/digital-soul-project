import os
import shutil
from pathlib import Path
import re

def clean_filename(filename):
    """Extract useful info from original filenames"""
    # Extract the numeric pattern if it exists
    match = re.search(r'M\d+_S\d+_G\d+', filename)
    if match:
        return match.group(0)
    return filename.split('_')[0]  # Fallback to first segment

def setup_gpt_dataset(input_dir, output_dir):
    """
    Organize audio files for GPT-SoVits training
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    wavs_dir = output_path / "wavs"
    
    # Create directory structure
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(output_path / "filelists", exist_ok=True)
    
    # Process files
    processed_files = []
    for idx, file_path in enumerate(sorted(input_path.glob("*.wav"))):
        # Create new standardized filename
        info = clean_filename(file_path.stem)
        new_name = f"2b_{info}_{idx:04d}.wav"
        
        # Copy file
        shutil.copy2(file_path, wavs_dir / new_name)
        processed_files.append(new_name)
        print(f"Processed: {file_path.name} -> {new_name}")
    
    # Create basic train/val split (90/10)
    split_idx = int(len(processed_files) * 0.9)
    train_files = processed_files[:split_idx]
    val_files = processed_files[split_idx:]
    
    # Write placeholder filelists
    def write_filelist(files, filename):
        with open(output_path / "filelists" / filename, 'w', encoding='utf-8') as f:
            for wav in files:
                # Format: wav_path|text|phonemes
                f.write(f"wavs/{wav}|||\n")
    
    write_filelist(train_files, "train.list")
    write_filelist(val_files, "val.list")
    
    print(f"\nDataset prepared in {output_dir}")
    print(f"Total files: {len(processed_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print("\nNote: Text and phoneme fields are empty and need to be filled")

if __name__ == "__main__":
    setup_gpt_dataset(
        input_dir=r"C:\Users\EVO\Documents\AI\GPT-SoVits\training_data\cleaned",
        output_dir=r"C:\Users\EVO\Documents\AI\GPT-SoVits\data\2b"
    )