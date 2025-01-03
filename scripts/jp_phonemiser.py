import os
from pathlib import Path
from pyopenjtalk import g2p
import jaconv
from tqdm import tqdm

def japanese_to_phonemes(text):
    """Convert Japanese text to phonemes using OpenJTalk"""
    try:
        # Convert half-width to full-width characters
        text = jaconv.h2z(text)
        # Get phonemes
        phonemes = g2p(text)
        # Clean up phonemes according to GPT-SoVits requirements
        phonemes = phonemes.replace('pau', ',').replace(' ', '')
        return phonemes
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error: {str(e)}")
        return ""

def process_filelist(data_dir, filename):
    """Process a single filelist file"""
    filepath = Path(data_dir) / "filelists" / filename
    new_lines = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"\nProcessing {filename}...")
    for line in tqdm(lines):
        parts = line.strip().split('|')
        if len(parts) >= 4:  # New format has 4 parts
            wav_path, speaker, lang, text = parts[:4]
            if text:  # Only process if there's text
                phonemes = japanese_to_phonemes(text)
                new_lines.append(f"{wav_path}|{speaker}|{lang}|{text}|{phonemes}\n")
            else:
                new_lines.append(line)
        else:
            print(f"Skipping malformed line: {line}")
            new_lines.append(line)
            
    # Write updated file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def add_phonemes(data_dir):
    """Add phonemes to all filelists"""
    data_path = Path(data_dir)
    
    # Process both train and validation lists
    for filename in ["train.list", "val.list"]:
        process_filelist(data_path, filename)
        
    print("\nPhoneme generation complete!")

def restore_and_update_filelist(data_dir, filename):
    """Restore from backup and add language tag"""
    filepath = Path(data_dir) / "filelists" / filename
    backup_path = Path(data_dir) / "filelists" / f"{filename}.backup"
    
    # Read backup
    with open(backup_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 3:  # Backup format has wav|text|phoneme
            wav_path, text, phoneme = parts[:3]
            # Add speaker and language tags
            new_line = f"{wav_path}|2b|ja|{text}|{phoneme}\n"
            new_lines.append(new_line)
    
    # Write updated file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    data_dir = r"C:\Users\EVO\Documents\AI\GPT-SoVits\data\2b"
    
    # Restore from backups first
    restore_and_update_filelist(data_dir, "train.list")
    restore_and_update_filelist(data_dir, "val.list")
    
    print("Files restored and updated with language tags!")