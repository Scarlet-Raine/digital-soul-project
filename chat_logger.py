import json
import time
import os
from datetime import datetime

class ChatLogger:
    def __init__(self):
        self.log_dir = os.path.join("audio", "out")  # Changed to audio/out
        self.current_conversation = []
        self.conversation_id = None
        self.ensure_directories()

    def ensure_directories(self):
        if not os.path.exists(self.log_dir):
            print(f"Error: Directory {self.log_dir} does not exist")
            return

    def start_new_conversation(self):
        # Save previous conversation if exists
        if self.current_conversation:
            self.save_current_conversation()
        
        # Start new conversation
        self.conversation_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_conversation = []

    def log_interaction(self, user_input, ai_response, audio_file=None):
        if not self.conversation_id:
            self.start_new_conversation()
            
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_input': user_input,
            'ai_response': ai_response,
            'audio_file': audio_file
        }
        
        self.current_conversation.append(log_entry)
        self.save_current_conversation()
        
        return log_entry

    def save_current_conversation(self):
        if not self.current_conversation:
            return
            
        log_path = os.path.join(self.log_dir, f"conversation_{self.conversation_id}.json")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'conversation_id': self.conversation_id,
                'start_time': self.current_conversation[0]['timestamp'],
                'end_time': self.current_conversation[-1]['timestamp'],
                'messages': self.current_conversation
            }, f, indent=2)

    def get_conversation_context(self, limit=5):
        """Get recent messages from current conversation"""
        return self.current_conversation[-limit:] if self.current_conversation else []