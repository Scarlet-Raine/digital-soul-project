python rvc_cli.py tts --tts_file "C:\Users\EVO\Documents\AI\rvc-cli\tts.txt" --tts_text "This is a test of the TTS to RVC pipeline." --tts_voice "en-US" --pth_path "C:\Users\EVO\Documents\AI\models\2BJP.pth" --input_path r"C:\Users\EVO\Documents\AI\audio\2b_base2.wav" --index_path "C:\Users\EVO\Documents\AI\models\2BJP.index" --output_tts_path "tts_output.wav" --output_rvc_path "rvc_output.wav"




python rvc_inf_cli.py infer --input_path "C:\Users\EVO\Documents\AI\StyleTTS2\result.wav" --output_path "C:\Users\EVO\Documents\AI\audio\out\out.wav" --pth_path "C:\Users\EVO\Documents\AI\models\2BJP.pth" --index_path "C:\Users\EVO\Documents\AI\models\2BJP.index" --pitch "2" --protect "0.49" --filter_radius "7" --clean_audio "true"