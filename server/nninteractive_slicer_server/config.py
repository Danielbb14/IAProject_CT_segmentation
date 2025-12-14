import os
from pathlib import Path
from dotenv import load_dotenv

# Path to the server directory (parent of the directory containing this config file)
# config.py is in server/nninteractive_slicer_server/
# so .env should be in server/
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env file if it exists
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# nnU-Net Configuration
# We use getenv to allow overriding via .env or shell environment variables
# Defaults match the previous hardcoded values to ensure it works out-of-the-box if .env is missing
NNUNET_RESULTS = os.getenv("NNUNET_RESULTS", "/home/moriarty_d/projects/nnunet-bbox/nnunet_results")

# Model Configuration
DATASET_ID = os.getenv("NNUNET_DATASET_ID", "Dataset002_axialBbox")
CONFIG = os.getenv("NNUNET_CONFIG", "3d_fullres")
FOLD = os.getenv("NNUNET_FOLD", "0")

# Point Model Configuration
POINT_DATASET_ID = os.getenv("NNUNET_POINT_DATASET_ID", "Dataset999_middleClick") # Example default
POINT_CONFIG = os.getenv("NNUNET_POINT_CONFIG", "3d_fullres")

# Server Configuration
ISAC_SERVER_URL = os.getenv("ISAC_SERVER_URL", "http://127.0.0.1:8000/IsacModelPredict")
ISAC_POINT_SERVER_URL = os.getenv("ISAC_POINT_SERVER_URL", "http://127.0.0.1:8000/IsacModelPredictPoint")
