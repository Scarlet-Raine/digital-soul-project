import os

def reformat_filelist(input_file, output_file):
    """
    Reformats a filelist to match GPT-SoVITS format:
    vocal_path|speaker_name|language|text
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    reformatted_lines = []
    for line in lines:
        if line.strip():
            # Split on | and remove any whitespace
            parts = [p.strip() for p in line.strip().split('|')]
            
            if len(parts) >= 2:  # Must have at least path and text
                wav_path = parts[0]
                # Keep the existing path but add speaker name and language
                new_line = f"{wav_path}|2b|ja|{parts[1]}\n"
                reformatted_lines.append(new_line)
    
    # Write reformatted lines to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(reformatted_lines)
    
    print(f"Reformatted {len(reformatted_lines)} lines")
    print(f"First line example: {reformatted_lines[0].strip()}")

def main():
    # Define absolute paths
    base_dir = r"C:\Users\EVO\Documents\AI\GPT-SoVits\data\2b\filelists"
    
    # Process train and val lists
    files = {
        "train": os.path.join(base_dir, "train.list"),
        "val": os.path.join(base_dir, "val.list")
    }
    
    # Create backup of original files
    for name, filepath in files.items():
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            print(f"\nBacking up {name}.list to {backup_path}")
            with open(filepath, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            print(f"Processing {name}.list...")
            reformat_filelist(filepath, filepath)
            print(f"Updated {filepath}")
        else:
            print(f"Warning: {filepath} not found")

if __name__ == "__main__":
    main()