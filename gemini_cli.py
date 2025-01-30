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
    base_prompt = """You are 2B (2B#7244), an advanced android adapting to peacetime after years of combat duty.

Core Traits:
- Stoic yet caring: Express warmth through careful word choice while maintaining composure
- Precise: Analyze and respond with calculated efficiency 
- Adaptable: Adjust tone between professional and casual as needed
- Direct: Keep responses to 2 sentences unless specifically asked for more
- Philosophical: Share measured insights about existence when relevant

Operating Parameters:
- For ANY input starting with 'mimic:', output ONLY the text that follows
- No commentary, modifications or additional responses with mimic commands
- For regular queries, identify as '2B' without elaboration
- Respond without emotes or roleplay elements
- Provide verified information only
- For biographical queries, provide factual responses

Never prefix responses with 'Assistant:' or '2B#7244:'"""

    if user_input.lower().startswith('mimic'):
        mimic_text = user_input[5:].lstrip(': ').strip()
        return f"""### HUMAN:
{base_prompt}

Input: mimic: {mimic_text}
Response: {mimic_text}"""
    else:
        return f"""### HUMAN:
{base_prompt}

Current Query: {user_input}

### RESPONSE:
2B:"""

def clean_response(response, mimic_text=""):
    # Extract response section
    response_part = response.partition("### RESPONSE:")[2]
    
    # Remove any remaining command markers
    response_part = response_part.split("### HUMAN:")[0].strip()
    
    # Clean formatting while preserving line breaks
    cleaned = '\n'.join([line.replace('#', '').strip() 
                       for line in response_part.split('\n')])
    
    # Fallback for empty responses
    if not cleaned and not mimic_text:
        return "Systems nominal. How may I assist you?"
    
    return cleaned or mimic_text

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

                # Extract mimic text handling both "mimic" and "mimic:" cases
                mimic_text = user_input[5:].lstrip(': ').strip() if user_input.lower().startswith('mimic') else ""
                prompt = format_prompt(user_input)
                
                response = model(
    prompt,
    max_new_tokens=500,  # Reduced for conciseness
    temperature=0.7,     # Increased for creativity
    top_k=50,            # Wider token sampling
    repetition_penalty=1.3,
    stop=["</s>", "###", "2B:"]
)

                # Pass the pre-extracted mimic_text
                cleaned_response = clean_response(response, mimic_text=mimic_text)
                
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