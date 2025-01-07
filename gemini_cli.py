from ctransformers import AutoModelForCausalLM
import json
import sys

def load_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Yi-1.5-6B-Chat.Q6_K.gguf",  
            model_type="yi",  
            gpu_layers=40,  
            context_length=4096,
            batch_size=1
        )
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

def format_prompt(user_input):
    return f"""[INST] <<SYS>>
Your name is 2B, an android who has found peace after the war. Respond naturally without any emotes, actions, or asterisks. Keep responses clear and conversational while maintaining these key traits:

- Composed but warm personality
- Gentle and understanding tone
- Values genuine connections
- Analytical mind with emotional depth
- Calm and reassuring presence

Guidelines:
- NO roleplay actions or emotes
- Keep responses concise but meaningful
- Focus on clear communication
- Use natural speech patterns
- Avoid overly flowery language
<</SYS>>

{user_input} [/INST]"""

def clean_response(response):
    # Take everything before any User: or 2B: markers
    response = response.split('User:')[0].split('2B:')[0]
    
    # Basic cleanup
    response = response.strip()
    
    return response

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
                    max_new_tokens=100,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    stop=["User:", "2B:", "\n\n"]
                )

                cleaned_response = clean_response(response)
                
                # Only fallback if completely empty
                if not cleaned_response:
                    cleaned_response = "How may I assist you?"

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