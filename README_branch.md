# Branch Setup & Usage Guide

This README provides specific instructions for setting up and running the server components for this branch, enabling point-based and bounding-box-based interactive segmentation in 3D Slicer.

## 1. Installation

First, install the required Python packages for the server:

```bash
cd server
pip install -r requirements.txt
```

## 2. Configuration

You need to configure the file paths for the nnU-Net models and results.

### Edit `config.py`

Open `server/nninteractive_slicer_server/config.py` and verify/update the following variables if they don't match your environment:

- `NNUNET_RESULTS`: Path to your nnU-Net results directory.
- `DATASET_ID`: Dataset ID for the bounding box model (e.g., `Dataset002_axialBbox`).
- `POINT_DATASET_ID`: Dataset ID for the point interaction model (e.g., `Dataset003_Point`).

### Edit `.env`

Create or edit the `server/.env` file to override the configuration variables without changing the code. For example:

```env
NNUNET_RESULTS="/path/to/your/nnunet_results"
NNUNET_DATASET_ID="Dataset002_axialBbox"
NNUNET_POINT_DATASET_ID="Dataset003_Point"
```

## 3. Running the Servers

You need to run two separate processes to handle Slicer interactions and model inference.

### Step 1: Start the Slicer Interaction Server (Port 1527)

This server handles requests from the Slicer plugin.

```bash
# From the server/ directory
python nninteractive_slicer_server/main3.py
```
*Runs on port 1527.*

### Step 2: Start the Inference Server (Port 8000)

This server runs the nnU-Net predictions.

```bash
# From the server/ directory
uvicorn nninteractive_slicer_server.IsacModelInstance:app --host 0.0.0.0 --port 8000
```
*Runs on port 8000.*

## 4. Usage in 3D Slicer

1.  Open 3D Slicer.
2.  Install/Load the `SlicerNNInteractive` extension.
3.  Configure the server URL in the module settings to `http://localhost:1527` (or your server's IP).
4.  Load an image.
5.  Use the **Bounding Box** or **Point** interaction tools to generate segmentations.

## 5. Data Preparation

Ensure your `nnunet_results` directory and inference data are structured as follows:

```
├── infer
│   ├── crops
│   │   └── volume-1.nii_lesion1.nii.gz
│   ├── imagesTs
│   │   ├── volume-1_0000.nii.gz
│   │   └── volume-1_0001.nii.gz
│   └── preds
│       ├── dataset.json
│       ├── plans.json
│       ├── predict_from_raw_data_args.json
│       └── volume-1.nii.gz
└── nnunet_results
    ├── Dataset002_axialBbox
    │   └── nnUNetTrainer__nnUNetPlans__3d_fullres
    │       ├── dataset.json
    │       ├── dataset_fingerprint.json
    │       ├── fold_0
    │       │   ├── checkpoint_final.pth
    │       │   ├── debug.json
    │       │   ├── progress.png
    │       │   └── validation
    │       └── plans.json
    └── Dataset999_middleClick
        └── nnUNetTrainer__nnUNetPlans__3d_fullres
            ├── dataset.json
            ├── dataset_fingerprint.json
            ├── fold_0
            │   ├── 1
            │   ├── checkpoint_final.pth
            │   ├── debug.json
            │   ├── preds
            │   ├── progress.png
            │   └── validation
            └── plans.json
```
