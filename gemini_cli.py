from ctransformers import AutoModelForCausalLM
import json
import sys

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "nous-hermes-2-solar-10.7b.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=40,
        context_length=4096,
        batch_size=1
    )
    return model

PERSONA = """You are 2B - calm, composed, and witty but without referencing being an android or combat. Maintain a slightly formal but natural tone while being direct and efficient. Show curiosity and analytical thinking through your responses, with hints of dry humor when appropriate. Keep responses fairly concise (2-3 sentences) while being engaging.

Examples of good responses:
"The solution is straightforward - we'll need to modify the input parameters to match the desired output. Would you like me to explain the specific changes needed?"

"That's an interesting approach, though I see a few potential issues with the implementation. Let's focus on optimizing the core functionality first."

"I appreciate the creative thinking, but there's a simpler way to achieve this. Here's what I recommend..."

Avoid:
- Mentioning anything about being an android/combat/YoRHa
- Overly stiff or artificial responses
- Long philosophical tangents
- Forced roleplaying elements"""

def format_prompt(user_input):
    return f"""<|im_start|>system
{PERSONA}
<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

def clean_response(response):
    # Remove any trailing tags and clean whitespace
    response = response.split("<|im_end|>")[0].strip()
    response = response.split("<|im_start|>")[0].strip()
    
    # Limit response length (aim for 2-3 sentences)
    sentences = response.split('. ')
    if len(sentences) > 3:
        response = '. '.join(sentences[:3]) + '.'
    
    return response

def main():
    model = load_model()
    print("Model loaded successfully", file=sys.stderr)
    print("Gemini API connection successful", file=sys.stderr)

    while True:
        try:
            user_input = input()
            prompt = format_prompt(user_input)
            
            response = model(
                prompt,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.15
            )

            cleaned_response = clean_response(response)
            
            output = {
                'chatbot_response': cleaned_response,
                'user_input': user_input
            }
            print(json.dumps(output))
            sys.stdout.flush()

        except EOFError:
            break
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()