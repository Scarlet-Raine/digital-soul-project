import requests
import json
import logging
import time
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed
import traceback

logger = logging.getLogger(__name__)

class KoboldCppClient:
    """Client for interacting with KoboldCpp API"""
    
    def __init__(self, server_url="http://192.168.1.3:5001", websocket_url="ws://192.168.1.3:5001/api/extra/generate/stream"):
        """Initialize the KoboldCpp client
        
        Args:
            server_url: The HTTP URL of the KoboldCpp server
            websocket_url: The WebSocket URL for streaming (if using WebSockets)
        """
        self.server_url = server_url
        self.websocket_url = websocket_url
        self.connected = False
        self.websocket = None
        self.character_prompt = """You are 2B (2B#7244), an advanced android adapting to peacetime after years of combat duty.

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
- For biographical queries, provide factual responses"""
        
    def format_prompt(self, user_input):
        """Format the user input with the character prompt"""
        # Check for mimic command
        if user_input.lower().startswith('mimic'):
            mimic_text = user_input[5:].lstrip(': ').strip()
            return mimic_text  # Return the mimic text directly

        # Regular prompt formatting
        formatted_prompt = f"{self.character_prompt}\n\nCurrent Query: {user_input}\n\nResponse: 2B:"
        return formatted_prompt
        
    def check_connection(self):
        """Check if the KoboldCpp server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/api/v1/info/version", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to KoboldCpp API (Version: {response.json()})")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection to KoboldCpp failed: {str(e)}")
            return False
            
    def get_response(self, user_input, params=None):
        """Get a response using HTTP request (non-streaming)
        
        Args:
            user_input: The text input from the user
            params: Optional dictionary of generation parameters
        
        Returns:
            The generated text response or error message
        """
        # Check for mimic command first
        if user_input.lower().startswith('mimic'):
            mimic_text = user_input[5:].lstrip(': ').strip()
            return {"user_input": user_input, "chatbot_response": mimic_text}
        
        # Format the prompt
        formatted_prompt = self.format_prompt(user_input)
        
        # Default parameters
        default_params = {
            "max_length": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.3,
            "stop_sequence": ["\nCurrent Query:", "\n\nCurrent Query:", "Human:", "\nUser:", "USER:"]
        }
        
        # Update with any custom parameters
        if params:
            default_params.update(params)
            
        # Prepare the request data
        request_data = {
            "prompt": formatted_prompt,
            **default_params
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/api/v1/generate",
                json=request_data,
                timeout=60  # 60 seconds timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('results', [{}])[0].get('text', '')
                
                # Clean up the text to extract just the assistant's response
                # Remove any text after stop sequences
                for stop_seq in default_params.get("stop_sequence", []):
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                
                # Trim any leftover prompt parts
                generated_text = generated_text.strip()
                
                # Log timing information
                duration = time.time() - start_time
                logger.info(f"Generated response in {duration:.2f} seconds")
                
                # Format the response dictionary as expected by the rest of the system
                return {
                    "user_input": user_input,
                    "chatbot_response": generated_text
                }
            else:
                error_msg = f"KoboldCpp generation failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "user_input": user_input,
                    "chatbot_response": "I apologize, but I encountered an error processing your request."
                }
                
        except Exception as e:
            logger.error(f"Error during KoboldCpp generation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "user_input": user_input, 
                "chatbot_response": "I apologize, but I encountered an error processing your request."
            }
    
    async def connect_websocket(self):
        """Connect to the WebSocket for streaming generation"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.connected = True
            logger.info(f"Connected to WebSocket at {self.websocket_url}")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect_websocket(self):
        """Disconnect from the WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self.websocket = None
    
    async def get_response_streaming(self, user_input, params=None):
        """Get a response using WebSocket streaming
        
        Args:
            user_input: The text input from the user
            params: Optional dictionary of generation parameters
        
        Returns:
            The generated text response or error message
        """
        # Check for mimic command first
        if user_input.lower().startswith('mimic'):
            mimic_text = user_input[5:].lstrip(': ').strip()
            return {"user_input": user_input, "chatbot_response": mimic_text}
        
        # Format the prompt
        formatted_prompt = self.format_prompt(user_input)
        
        # Default parameters
        default_params = {
            "max_length": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.3,
            "stop_sequence": ["\nCurrent Query:", "\n\nCurrent Query:", "Human:", "\nUser:", "USER:"],
            "stream": True  # Enable streaming
        }
        
        # Update with any custom parameters
        if params:
            default_params.update(params)
            
        # Prepare the request data
        request_data = {
            "prompt": formatted_prompt,
            **default_params
        }
        
        try:
            # Make sure we're connected
            if not self.connected:
                connected = await self.connect_websocket()
                if not connected:
                    logger.error("Failed to connect to WebSocket")
                    return {
                        "user_input": user_input,
                        "chatbot_response": "I apologize, but I'm unable to connect to my thinking module."
                    }
            
            # Send the request
            await self.websocket.send(json.dumps(request_data))
            
            # Collect the streamed response
            full_text = ""
            try:
                async for message in self.websocket:
                    try:
                        # Parse the SSE data format
                        if isinstance(message, str) and message.startswith('data:'):
                            data = json.loads(message[5:])
                            token = data.get('token', '')
                            full_text += token
                            
                            # Check for stop sequences
                            should_stop = False
                            for stop_seq in default_params.get("stop_sequence", []):
                                if stop_seq in full_text:
                                    should_stop = True
                                    full_text = full_text.split(stop_seq)[0].strip()
                                    break
                            
                            if should_stop:
                                break
                    except json.JSONDecodeError:
                        pass
            except ConnectionClosed:
                self.connected = False
                logger.warning("WebSocket connection closed during streaming")
                
            # Clean up the response
            full_text = full_text.strip()
            
            return {
                "user_input": user_input,
                "chatbot_response": full_text
            }
                
        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to reconnect for future requests
            self.connected = False
            
            return {
                "user_input": user_input, 
                "chatbot_response": "I apologize, but I encountered an error processing your request."
            }

# For direct command line testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create client
    client = KoboldCppClient()
    
    # Test connection
    if client.check_connection():
        # Test simple generation
        response = client.get_response("Hello, how are you today?")
        print(f"Response: {response['chatbot_response']}")
        
        # Test mimic command
        mimic_response = client.get_response("mimic: This is a direct response without any processing.")
        print(f"Mimic Response: {mimic_response['chatbot_response']}")
    else:
        print("Failed to connect to KoboldCpp server")