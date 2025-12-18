import json
import os
import sys

def load_config():
    # Determine the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # The config.json is expected to be in the parent 'server' directory 
    # relative to 'nninteractive_slicer_server' where this script lives,  
    # or just in the parent of the package.
    # We look up one level from this file (which is in nninteractive_slicer_server)
    # to find 'server/config.json'
    
    config_path = os.path.join(current_dir, "..", "config.json")
    
    # Normalize path
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using defaults might fail.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

# Load config once when module is imported
config = load_config()
