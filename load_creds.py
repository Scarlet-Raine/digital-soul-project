# load_creds.py
from google.oauth2.credentials import Credentials
import os
import json

def load_creds():
    """Load Google API credentials"""
    try:
        # Try to load the API key from client_secret.json
        creds_path = "client_secret.json"
        
        if not os.path.exists(creds_path):
            raise FileNotFoundError(f"Credentials file not found at {creds_path}")
            
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            
        if 'api_key' in creds_data:
            return creds_data['api_key']
        else:
            raise KeyError("No API key found in credentials file")
            
    except Exception as e:
        print(f"Error loading credentials: {str(e)}")
        raise