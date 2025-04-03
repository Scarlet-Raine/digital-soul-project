import json
import time
import threading
import queue
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_client.log")
    ]
)
logger = logging.getLogger(__name__)

# A simple mock function to generate responses
def mock_generate(user_input):
    """Generate a simple mock response for testing"""
    logger.info(f"Mock processing: {user_input[:50]}...")
    time.sleep(1)  # Simulate processing time
    return {
        "chatbot_response": f"Mock response to: {user_input[:50]}...",
        "user_input": user_input
    }

def run_llm_service(input_queue, output_deque, server_url, shutdown_event):
    """
    Run a simplified LLM service that just generates mock responses
    for testing the pipeline without an actual LLM server
    
    Args:
        input_queue: Queue for input messages
        output_deque: Deque for output messages
        server_url: Not used in mock version
        shutdown_event: Event to signal shutdown
    """
    logger.info(f"Starting simplified mock LLM service")
    
    # Main processing loop
    while not shutdown_event.is_set():
        try:
            # Check for input with timeout
            try:
                user_input = input_queue.get(timeout=0.5)
                logger.info(f"Processing input: {user_input[:50]}...")
                
                # Generate a mock response
                response = mock_generate(user_input)
                
                # Add to output deque
                output_deque.append(response)
                logger.info(f"Added response to output deque: {response['chatbot_response'][:50]}...")
                print(f"\033[95m[{time.strftime('%H:%M:%S')}] LLM mock response: {response['chatbot_response']}\033[0m")
                
            except queue.Empty:
                # No input, just continue
                time.sleep(0.1)
                continue
                
        except Exception as e:
            logger.error(f"Error in mock LLM service: {str(e)}")
            try:
                # Add error response to output deque
                error_response = {
                    "error": str(e),
                    "chatbot_response": "I encountered an error processing your request.",
                    "user_input": user_input if 'user_input' in locals() else "unknown"
                }
                output_deque.append(error_response)
            except:
                pass
            
            # Short delay to prevent CPU spinning on repeated errors
            time.sleep(1)
            
    logger.info("Mock LLM service shutting down")
    print(f"\033[93m[{time.strftime('%H:%M:%S')}] Mock LLM service shutting down\033[0m")