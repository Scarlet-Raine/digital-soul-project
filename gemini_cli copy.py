import sys
import google.generativeai as palm
from google.generativeai import types
from load_creds import load_creds
import json
from google.api_core.timeout import TimeToDeadlineTimeout
import google.generativeai as genai  # Note: changed import name
from load_creds import load_creds

# --- Configuration ---
api_key = load_creds()
genai.configure(api_key=api_key)
MODEL = 'gemini-1.5-flash-8b'
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024

def generate_text(prompt):
    """Sends a prompt to the Gemini API and returns the generated text."""
    model = genai.GenerativeModel(MODEL)

    response = model.generate_content(
        prompt,
        generation_config={
            'temperature': TEMPERATURE,
            'max_output_tokens': MAX_OUTPUT_TOKENS,
        }
    )

    return response.text

# --- Modules ---

def generate_text(prompt):
    """Sends a prompt to the Gemini API and returns the generated text."""
    try:
        print(f"Generating response for: {prompt}", file=sys.stderr)
        model = genai.GenerativeModel(MODEL)

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': TEMPERATURE,
                'max_output_tokens': MAX_OUTPUT_TOKENS,
            }
        )
        print(f"Got response from API: {response.text}", file=sys.stderr)
        return response.text
    except Exception as e:
        print(f"Error in generate_text: {str(e)}", file=sys.stderr)
        raise

# --- Main Loop ---


def main():
    try:
        # Test API access immediately
        model = palm.GenerativeModel(MODEL)
        test_response = model.generate_content("test")
        print("Gemini API connection successful", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize Gemini API: {str(e)}", file=sys.stderr)
        return

    while True:
        try:
            if len(sys.argv) > 1:
                user_input = sys.argv[1]
                sys.argv = sys.argv[:1]
                print(f"Received input: {user_input}", file=sys.stderr)
            else:
                try:
                    user_input = input()
                except EOFError:
                    print("EOFError - Exiting...", file=sys.stderr)
                    break
                except KeyboardInterrupt:
                    print("KeyboardInterrupt - Exiting...", file=sys.stderr)
                    break

            if not user_input or user_input.lower() in ["quit", "exit"]:
                break

            try:
                response = generate_text(user_input)
                response = response.rstrip("\n")
                output = json.dumps({'chatbot_response': response})
                print(output)
                sys.stdout.flush()
            except Exception as e:
                print(f"Error generating response: {str(e)}", file=sys.stderr)
                # Still output a valid JSON response even on error
                print(json.dumps({'chatbot_response': f"Error: {str(e)}"}))
                sys.stdout.flush()

        except Exception as e:
            print(f"Error in main loop: {str(e)}", file=sys.stderr)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Critical error: {str(e)}", file=sys.stderr)