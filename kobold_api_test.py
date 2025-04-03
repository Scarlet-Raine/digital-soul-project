import requests
import json
import time

# Configure the KoboldCpp server address
KOBOLDCPP_URL = "http://192.168.1.3:5001"  # Change this to your secondary machine's IP and port

def test_connection():
    """Test basic connection to KoboldCpp server"""
    try:
        # Check version
        version_response = requests.get(f"{KOBOLDCPP_URL}/api/v1/info/version")
        extra_version = requests.get(f"{KOBOLDCPP_URL}/api/extra/version")
        
        # Get model info
        model_response = requests.get(f"{KOBOLDCPP_URL}/api/v1/model")
        
        # Get max context length
        max_context = requests.get(f"{KOBOLDCPP_URL}/api/v1/config/max_context_length")
        
        print("=== Connection Test Results ===")
        print(f"API Version: {version_response.json()}")
        print(f"KoboldCpp Version: {extra_version.json()}")
        print(f"Model: {model_response.json()}")
        print(f"Max Context Length: {max_context.json()}")
        print("Connection successful!")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def test_generation(prompt="Hello, my name is", max_length=100):
    """Test text generation with KoboldCpp"""
    try:
        # Prepare the generation request
        generation_data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.18
        }
        
        print(f"\n=== Testing Generation ===")
        print(f"Prompt: {prompt}")
        print("Generating...")
        
        start_time = time.time()
        response = requests.post(
            f"{KOBOLDCPP_URL}/api/v1/generate",
            json=generation_data
        )
        end_time = time.time()
        
        # Extract and display results
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('results', [{}])[0].get('text', 'No text generated')
            
            print(f"\nGenerated Text:\n{generated_text}")
            print(f"\nGeneration Time: {end_time - start_time:.2f} seconds")
            return True
        else:
            print(f"Generation failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Generation error: {e}")
        return False

def test_streaming(prompt="Once upon a time,", max_length=100):
    """Test streaming text generation with KoboldCpp"""
    try:
        # Prepare the generation request
        generation_data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.18,
            "stream": True  # Enable streaming
        }
        
        print(f"\n=== Testing Streaming Generation ===")
        print(f"Prompt: {prompt}")
        print("Generating (streaming)...")
        
        # Using the streaming endpoint
        response = requests.post(
            f"{KOBOLDCPP_URL}/api/extra/generate/stream",
            json=generation_data,
            stream=True
        )
        
        # Process the streamed response
        full_text = ""
        start_time = time.time()
        for line in response.iter_lines():
            if line:
                # SSE format: data: {...}
                line_text = line.decode('utf-8')
                if line_text.startswith('data:'):
                    try:
                        data = json.loads(line_text[5:])
                        token = data.get('token', '')
                        full_text += token
                        print(token, end='', flush=True)
                    except json.JSONDecodeError:
                        pass
        
        end_time = time.time()
        print(f"\n\nFull Text (Streaming):\n{full_text}")
        print(f"Streaming Generation Time: {end_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Streaming error: {e}")
        return False

def format_prompt_for_chatbot(user_input, system_prompt=None):
    """Format a prompt suitable for chatbot interaction"""
    if system_prompt:
        formatted_prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    else:
        formatted_prompt = f"User: {user_input}\nAssistant:"
    return formatted_prompt

def test_chatbot_response(user_input, system_prompt=None):
    """Test chatbot-style response generation"""
    formatted_prompt = format_prompt_for_chatbot(user_input, system_prompt)
    
    print(f"\n=== Testing Chatbot Response ===")
    print(f"User Input: {user_input}")
    
    # Configure the generation parameters for chat
    generation_data = {
        "prompt": formatted_prompt,
        "max_length": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "stop_sequence": ["\nUser:", "\n\nUser:", "\nHuman:"]  # Stop generation at these sequences
    }
    
    try:
        response = requests.post(
            f"{KOBOLDCPP_URL}/api/v1/generate",
            json=generation_data
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('results', [{}])[0].get('text', 'No response generated')
            
            # Clean up the generated text to just get the assistant's response
            print(f"\nAssistant's Response:\n{generated_text.strip()}")
            return generated_text
        else:
            print(f"Chatbot response failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Chatbot response error: {e}")
        return None

if __name__ == "__main__":
    # Run tests
    if test_connection():
        test_generation()
        test_streaming()
        
        # Test chatbot functionality with a system prompt
        system_prompt = """You are a helpful assistant."""
        test_chatbot_response("Tell me a quick joke about programming.", system_prompt)
        
        # Test with character persona
        character_prompt = """You are 2B (2B#7244), an advanced android adapting to peacetime after years of combat duty.

Core Traits:
- Stoic yet caring: Express warmth through careful word choice while maintaining composure
- Precise: Analyze and respond with calculated efficiency 
- Adaptable: Adjust tone between professional and casual as needed
- Direct: Keep responses to 2 sentences unless specifically asked for more
- Philosophical: Share measured insights about existence when relevant"""
        
        test_chatbot_response("How are you doing today?", character_prompt)