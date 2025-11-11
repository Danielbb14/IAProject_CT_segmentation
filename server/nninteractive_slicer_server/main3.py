import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import gzip
from typing import List, Dict
import subprocess
import os
import tempfile
import nibabel as nib

app = FastAPI()

# Global storage
current_image = None
segmentation_history: List[Dict] = []  # Store all segmentations separately
next_segment_id = 1

class BBoxParams(BaseModel):
    outer_point_one: list[int]  # [x, y, z] - already reversed by plugin
    outer_point_two: list[int]  # [x, y, z] - already reversed by plugin
    positive_click: bool = True

def create_mock_segmentation(image_shape, bbox_coords, segment_id):
    """
    Create a mock segmentation - small ball/lesion in the bounding box center
    """
    # Create empty segmentation
    seg = np.zeros(image_shape, dtype=np.uint8)
    
    # Calculate bounding box center (coordinates already reversed by plugin)
    p1, p2 = np.array(bbox_coords[0]), np.array(bbox_coords[1])
    center = ((p1 + p2) / 2).astype(int)
    
    # Ensure center is within image bounds
    center = np.clip(center, 0, np.array(image_shape) - 1)
    
    # Create a small sphere around the center (radius varies by segment_id for visual distinction)
    radius = 8 + (segment_id % 3) * 4  # Vary radius: 8, 12, 16, then repeat
    x, y, z = np.ogrid[:image_shape[0], :image_shape[1], :image_shape[2]]
    
    # Distance from center
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    # Create sphere
    seg[dist <= radius] = 1
    
    return seg

def run_nnunet_segmentation(image: np.ndarray, bbox_coords: List[List[int]], dataset_id: str, config: str, fold: str) -> np.ndarray | None:
    """
    Runs nnU-Net prediction on a given image and bounding box.

    Args:
        image: The input image as a NumPy array.
        bbox_coords: A list containing two points [p1, p2] that define the bounding box.
        dataset_id: The nnU-Net Dataset ID (e.g., "Dataset123_TaskName").
        config: The nnU-Net configuration (e.g., "3d_fullres").
        fold: The trained model fold to use for inference (e.g., "0").

    Returns:
        A NumPy array containing the segmentation mask, or None if an error occurs.
    """
    p1, p2 = np.array(bbox_coords[0]), np.array(bbox_coords[1])

    # Determine the bounding box slices, assuming (z, y, x) order
    z_min, y_min, x_min = np.min([p1, p2], axis=0)
    z_max, y_max, x_max = np.max([p1, p2], axis=0)

    # Create the bounding box channel
    bbox_channel = np.zeros(image.shape, dtype=np.uint8)
    bbox_channel[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 1

    # Create temporary directories for nnU-Net input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        try:
            # 1. Save the current image and the bbox channel as NIfTI files
            image_path_0000 = os.path.join(input_dir, "image_0000.nii.gz")
            image_path_0001 = os.path.join(input_dir, "image_0001.nii.gz")
            
            affine = np.eye(4)  # Using identity affine

            nifti_img_0000 = nib.Nifti1Image(image, affine)
            nib.save(nifti_img_0000, image_path_0000)

            nifti_img_0001 = nib.Nifti1Image(bbox_channel, affine)
            nib.save(nifti_img_0001, image_path_0001)

            # 2. Construct and execute the nnU-Net command
            command = [
                "nnUNetv2_predict",
                "-i", input_dir,
                "-o", output_dir,
                "-d", dataset_id,
                "-c", config,
                "-f", fold,
                "--disable_tta"
            ]
            
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            print("nnU-Net stdout:", result.stdout)
            print("nnU-Net stderr:", result.stderr)

            # 3. Read and return the output segmentation
            output_path = os.path.join(output_dir, "image.nii.gz")
            if not os.path.exists(output_path):
                raise FileNotFoundError("nnU-Net did not produce an output file.")

            seg_img = nib.load(output_path)
            seg_array = seg_img.get_fdata().astype(np.uint8)
            return seg_array

        except subprocess.CalledProcessError as e:
            print(f"nnU-Net command failed with exit code {e.returncode}")
            print("Stderr:", e.stderr)
            return None
        except Exception as e:
            print(f"An unexpected error occurred during nnU-Net prediction: {e}")
            return None

def combine_all_segmentations(image_shape):
    """
    Combine all stored segmentations into one final mask
    """
    if not segmentation_history:
        return np.zeros(image_shape, dtype=np.uint8)
    
    # Start with empty
    combined = np.zeros(image_shape, dtype=np.uint8)
    
    # Add all positive segmentations, subtract negative ones
    for seg_data in segmentation_history:
        if seg_data['positive']:
            combined = np.logical_or(combined, seg_data['mask']).astype(np.uint8)
        else:
            combined = np.logical_and(combined, ~seg_data['mask']).astype(np.uint8)
    
    return combined

def pack_and_compress_segmentation(seg_array):
    """
    Pack segmentation into binary format and compress (same format as original)
    """
    # Convert to bool and pack bits
    seg_bool = seg_array.astype(bool)
    packed = np.packbits(seg_bool, axis=None)
    
    # Compress with gzip
    compressed = gzip.compress(packed.tobytes())
    return compressed

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and store the CT image
    """
    global current_image
    
    # Read and load the numpy array
    file_bytes = await file.read()
    current_image = np.load(io.BytesIO(file_bytes))
    
    print(f"Image uploaded with shape: {current_image.shape}")
    return {"status": "ok"}

@app.post("/upload_segment")
async def upload_segment(file: UploadFile = File(...)):
    """
    Upload current segmentation state from plugin
    This could be used to initialize our history, but for now we'll just acknowledge it
    """
    global current_image, segmentation_history
    
    # Check if image is uploaded first
    if current_image is None:
        return {"status": "error", "message": "No image uploaded"}
    
    # Read, decompress and load the numpy array
    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)
    current_segment = np.load(io.BytesIO(decompressed))
    
    # If plugin sends an empty segment, clear our history
    if np.sum(current_segment) == 0:
        segmentation_history.clear()
        print("Cleared segmentation history (empty segment received)")
    else:
        # Plugin sent existing segmentation - we could add it to history
        # For now, just acknowledge it
        print(f"Received existing segment with {np.sum(current_segment)} voxels")
    
    return {"status": "ok"}

@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    """
    Add a new bounding box segmentation to our collection
    """
    global current_image, segmentation_history, next_segment_id
    
    # Check if image is uploaded
    if current_image is None:
        return {"status": "error", "message": "No image uploaded"}
    
    print(f"Received bbox (reversed): {params.outer_point_one} to {params.outer_point_two}")
    print(f"Positive click: {params.positive_click}")
    
    # Create new segmentation for this bounding box using nnU-Net
    DATASET_ID = "Dataset999_middleClick"
    CONFIG = "3d_fullres"
    FOLD = "0"

    bbox = [params.outer_point_one, params.outer_point_two]
    
    new_seg = run_nnunet_segmentation(current_image, bbox, DATASET_ID, CONFIG, FOLD)

    if new_seg is None:
        return {"status": "error", "message": "nnU-Net prediction failed."}
    
    # Store this segmentation in our history
    segmentation_history.append({
        'id': next_segment_id,
        'mask': new_seg,
        'positive': params.positive_click,
        'bbox': [params.outer_point_one, params.outer_point_two],
        'voxel_count': int(np.sum(new_seg))
    })
    
    print(f"Added segment {next_segment_id} ({'positive' if params.positive_click else 'negative'}) with {np.sum(new_seg)} voxels")
    next_segment_id += 1
    
    # Combine all segmentations
    combined_seg = combine_all_segmentations(current_image.shape)
    
    # Pack and compress the result
    compressed_result = pack_and_compress_segmentation(combined_seg)
    
    print(f"Returning combined segmentation with {np.sum(combined_seg)} total voxels")
    print(f"Total segments in history: {len(segmentation_history)}")
    
    return Response(
        content=compressed_result,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )

@app.get("/status")
async def get_status():
    """
    Check server status and current state info
    """
    global current_image, segmentation_history
    
    return {
        "status": "ready", 
        "image_loaded": current_image is not None,
        "image_shape": list(current_image.shape) if current_image is not None else None,
        "total_segments": len(segmentation_history),
        "segments": [
            {
                "id": seg["id"], 
                "positive": seg["positive"], 
                "voxel_count": seg["voxel_count"]
            } 
            for seg in segmentation_history
        ]
    }

@app.post("/clear_history")
async def clear_history():
    """
    Clear all segmentation history
    """
    global segmentation_history, next_segment_id
    
    segmentation_history.clear()
    next_segment_id = 1
    
    print("Cleared all segmentation history")
    return {"status": "ok", "message": "History cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1527)