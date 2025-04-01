import os
import sys
import nltk
import ssl
from pathlib import Path

def download_all_required():
    """Downloads all required NLTK resources including the extended set needed."""
    
    # Create an NLTK data directory in the project
    nltk_data_dir = Path.home() / "AppData" / "Roaming" / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ['NLTK_DATA'] = str(nltk_data_dir)
    
    print(f"Installing NLTK data to: {nltk_data_dir}")
    
    # Extended list of resources
    resources = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'cmudict',
        'words',
        'tagsets',
        'universal_tagset',
        'brown',
        'treebank'
    ]
    
    # Download resources
    for resource in resources:
        try:
            print(f"\nDownloading {resource}...")
            nltk.download(resource, download_dir=str(nltk_data_dir), quiet=False)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
            
    # Special handling for punkt_tab
    punkt_dir = nltk_data_dir / "tokenizers" / "punkt_tab" / "english"
    punkt_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify installation with actual use
    print("\nVerifying installations...")
    
    try:
        test_text = "This is a test sentence."
        tokens = nltk.word_tokenize(test_text)
        tags = nltk.pos_tag(tokens)
        print(f"\nTest successful! Sample output:")
        print(f"Tokens: {tokens}")
        print(f"Tags: {tags}")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    print("Starting complete NLTK resource installation...")
    download_all_required()
    print("\nSetup complete!")