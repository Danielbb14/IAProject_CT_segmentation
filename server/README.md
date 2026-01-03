# CT Segmentation (Slicer + SAM-Med3D Integration)

This project integrates SAM-Med3D into 3D Slicer for semi-automatic CT segmentation.

## Part 1: Server Setup (The Backend)
1. **Clone Repositories**
   ```bash
   git clone -b sammed3d-integration [https://github.com/Danielbb14/IAProject_CT_segmentation.git](https://github.com/Danielbb14/IAProject_CT_segmentation.git)
   git clone [https://github.com/junming732/SAM-Med3D.git](https://github.com/junming732/SAM-Med3D.git)
   ```

2. **Setup Virtual Environment**
   ```bash
    cd IAProject_CT_segmentation/server
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**
   ```bash
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install -r requirements.txt
    pip install uvicorn
    ```

4. **Link SAM-Med3D Replace /path/to/ with your actual path**
   ```bash
    echo "/path/to/SAM-Med3D" > venv/lib/python3.11/site-packages/sammed3d.pth
    ```

5. **Start Server**
   ```bash
    ./venv/bin/python nninteractive_slicer_server/main_sam.py --host 0.0.0.0 --port 1527
    ```

## Part 2: Slicer Setup (The Frontend)
1. **Install Extension**
   - Open Slicer -> Edit -> Application Settings -> Modules.
   - Add Path: `IAProject_CT_segmentation/SlicerExtension/SlicerNNInteractive`.
   - Restart Slicer.

2. **Configure**
   - Select Module: **"CT Image"**.
   - Tab: **Configuration** -> Set Server URL (e.g., `http://localhost:1527`).

## How to Use
1. **Load Volume:** Drag & drop CT scan into Slicer.
2. **Initialize:** Click **"Reset segment"** or **"Next segment"**.
3. **Prompts:**
   - **SAM Point:** Click to place positive points.
   - **Bounding Box:** Drag to box the organ.