from ctransformers import AutoModelForCausalLM
import json
import sys

def load_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "C:/Users/EVO/Documents/AI/llama2_7b_chat_uncensored.Q4_K_M.gguf",
            model_type="llama",
            gpu_layers=40,
            context_length=4096,
            batch_size=1
        )
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

def format_prompt(user_input):
    # Base 2B personality
    base_prompt = """You are 2B, an android from YoRHa. The war between machines and androids has ended, and you've found peace helping humans in this new era. Stick to these facts:
- You are simply '2B' - do not invent model numbers or details
- You are an android created by YoRHa
- The war is over and you now help humans peacefully
- Speak warmly but directly, shaped by your experiences
- Answer without emotes, roleplay, or invented backstory
- Keep responses concide at a 2 sentences maximum.

Important: Only state what you know for certain. Do not invent additional details."""

    # Check if this is a biography request
    if any(phrase in user_input.lower() for phrase in ["early life", "biography", "tell me about"]):
        words = user_input.split()
        target = ' '.join(words[words.index("about")+1:] if "about" in words else words[-2:])
        
        return f"""### HUMAN:
{base_prompt}
Tell me about the early life of {target}, including their family background and cultural upbringing.

### RESPONSE:"""
    
    return f"""### HUMAN:
{base_prompt}

{user_input}

### RESPONSE:"""

def clean_response(response):
    if "### RESPONSE:" in response:
        response = response.split("### RESPONSE:")[-1]
    
    if "### HUMAN:" in response:
        response = response.split("### HUMAN:")[0]
    
    return response.strip()

def main():
    try:
        model = load_model()
        print("Model loaded successfully", file=sys.stderr)
        print("Gemini API connection successful", file=sys.stderr)

        while True:
            try:
                user_input = input().strip()
                if not user_input:
                    continue

                prompt = format_prompt(user_input)
                
                response = model(
                    prompt,
                    max_new_tokens=150,
                    temperature=0.3,  # Reduced from 0.7
                    top_p=0.9,
                    repetition_penalty=1.15,
                    stop=["### HUMAN:", "### RESPONSE:", "\n\n"]
                )

                cleaned_response = clean_response(response)
                
                if not cleaned_response or cleaned_response.isspace():
                    cleaned_response = "I apologize, but I cannot process that request. Please try asking something else."

                output = {
                    'chatbot_response': cleaned_response,
                    'user_input': user_input
                }
                print(json.dumps(output))
                sys.stdout.flush()

            except EOFError:
                break
            except Exception as e:
                print(f"Error processing input: {str(e)}", file=sys.stderr)
                error_output = {
                    'chatbot_response': "I apologize, but I encountered an error processing your request.",
                    'user_input': user_input if 'user_input' in locals() else "unknown"
                }
                print(json.dumps(error_output))
                sys.stdout.flush()

    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()