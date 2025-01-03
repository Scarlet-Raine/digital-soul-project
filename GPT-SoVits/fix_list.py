import os

list_path = os.path.join(os.getcwd(), "data/ch/filelists/fix_list.py")
with open(r"C:\Users\EVO\Documents\AI\GPT-SoVits\data\2b\filelists\val.list", 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    parts = line.strip().split('|')
    fixed_line = f"{parts[0]}|{parts[1]}|{parts[2]}|{parts[3]}\n"
    fixed_lines.append(fixed_line)

with open(r"C:\Users\EVO\Documents\AI\GPT-SoVits\data\2b\filelists\val_fixed.list", 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)