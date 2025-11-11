#!/bin/bash

# =====================================================================================
#  Startup Script for the 3D Slicer Interactive Segmentation Server
# =====================================================================================
#
#  This script sets the required environment variables for nnU-Net
#  and starts the FastAPI server.
#
# =====================================================================================

# --- Configure nnU-Net Environment Variables ---
# This path should point to the parent directory of your nnU-Net datasets
export nnUNet_results="/home/moriarty_d/projects/nnunet-oneclick/nnunet_results"
export nnUNet_preprocessed="/home/moriarty_d/projects/nnunet-oneclick/nnunet_preprocessed"
export nnUNet_raw="/home/moriarty_d/projects/nnunet-oneclick/nnunet_raw"


# --- Do not edit below this line ---
echo "==========================================="
echo "  Starting 3D Slicer FastAPI Server"
echo "==========================================="
echo "Using nnU-Net Results Path: $nnUNet_results"
echo "-------------------------------------------"

# Get the directory where this script is located to find the python script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the python script which starts the uvicorn server on port 1527
python3 "$SCRIPT_DIR/nninteractive_slicer_server/main3.py"