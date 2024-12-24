import speech_recognition as sr
import google.generativeai as palm
from google.generativeai import types
from load_creds import load_creds
import json

# --- Configuration ---
creds = load_creds()
palm.configure(credentials=creds)
MODEL = 'models/gemini-1.5-flash'
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024

# --- Modules ---

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")

        while True:
            try:
                # Listen for audio with a dynamic timeout
                recorded_audio = recognizer.listen(source, phrase_time_limit=2, timeout=None)  
                print("Processing...")
                break  # Exit the loop if audio is captured
            except sr.WaitTimeoutError:
                print("...")  # Indicate that it's still listening

    try:
        text = recognizer.recognize_google(recorded_audio, language="en-US")
        print("Decoded Text : {}".format(text))
        return text
    except Exception as ex:
        print(ex)
        return ""

def generate_text(prompt):
    """Sends a prompt to the Gemini API and returns the generated text."""
    model = palm.GenerativeModel(MODEL)
    response = model.generate_content(
        [prompt],
        generation_config=types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    )
    return response.text

# --- Main Loop ---

def main():
    while True:
        user_input = speech_to_text()
        if user_input.lower() in ["quit", "exit"]:
            break
        response = generate_text(user_input)
        
        # Remove the newline character from the response
        response = response.rstrip("\n")

        # Return the response as JSON 
        print(json.dumps({'chatbot_response': response}, separators=(',', ':'))) 


if __name__ == '__main__':
    main() 