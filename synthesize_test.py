import torch
from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.models.styletts2 import StyleTTS2
from TTS.utils.audio import AudioProcessor
from TTS.inference import AutoProcessor  # For the text processor

# Load the configuration file
model_path = r"C:\Users\EVO\Documents\AI\StyleTTS2-LJSpeech\Models\LJSpeech"
config_path = os.path.join(model_path, "config.yml")
config = StyleTTS2Config()
config.load_json(config_path)

# Load the model
model = StyleTTS2(config)
model.load_checkpoint(config, checkpoint_path=os.path.join(model_path, "epoch_2nd_00100.pth"))
model.to(device)  # Move the model to the appropriate device (CPU or GPU)

# Initialize the audio processor (if needed)
# Adjust parameters based on your model's configuration
audio_processor = AudioProcessor(
    sample_rate=config.audio.sample_rate,
    n_fft=config.audio.n_fft,
    win_length=config.audio.win_length,
    hop_length=config.audio.hop_length,
    n_mels=config.audio.n_mels,
    fmin=config.audio.fmin,
    fmax=config.audio.fmax,
)

# Initialize the text processor
processor = AutoProcessor.from_pretrained(model_path)

# Prepare the text
text = "Uncounted mages, wizards, warlocks, dragons, druids, illusionists, necromancers and such swear that they know how to control magic, bend it to their will, and force it to do what they wanted."

# --- Inference ---

# Tokenize the text
inputs = processor.text_to_sequence(text)
inputs = torch.tensor(inputs).long().unsqueeze(0).to(device)

# Get the output from the model
with torch.no_grad():
    outputs = model.inference(inputs)

# Convert the output to waveform (if necessary)
wav = audio_processor.inv_melspectrogram(outputs["mel_outputs"][0].T)

# Save the audio
output_path = r"C:\Users\EVO\Documents\AI\audio\out\output.wav"
audio_processor.save_wav(wav, output_path)