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

import requests
import numpy as np
from typing import List, Union

app = FastAPI()

# Global storage
current_image = None
segmentation_history: List[Dict] = []  # Store all segmentations separately
next_segment_id = 1

class BBoxParams(BaseModel):
    outer_point_one: list[int]  # [x, y, z] - already reversed by plugin
    outer_point_two: list[int]  # [x, y, z] - already reversed by plugin
    positive_click: bool = True



def call_isac_model_predict(
    image: np.ndarray,
    bbox_coords: List[List[int]],
    dataset_id: str,
    config: str,
    fold: str,
    server_url: str = "http://127.0.0.1:8000/IsacModelPredict"
) -> Union[np.ndarray, None]:
    """
    Calls the IsacModelPredict FastAPI endpoint and returns the segmentation result.

    Args:
        image: The input image as a NumPy array.
        bbox_coords: Bounding box coordinates [[p1], [p2]].
        dataset_id: nnU-Net dataset ID (e.g., "Dataset123_TaskName").
        config: nnU-Net configuration (e.g., "3d_fullres").
        fold: Model fold to use (e.g., "0").
        server_url: URL of the FastAPI endpoint.

    Returns:
        Segmentation mask as a NumPy array, or None if the request fails.
    """

    payload = {
        "image": image.tolist(),
        "bbox_coords": bbox_coords,
        "dataset_id": dataset_id,
        "config": config,
        "fold": fold
    }

    try:
        response = requests.post(server_url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success" and data.get("prediction") is not None:
            return np.array(data["prediction"], dtype=np.uint8)
        else:
            print("Server returned an error:", data)
            return None

    except requests.RequestException as e:
        print(f"Request failed: {e}")
    return None



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
    # IMPORTANT: Replace these with your actual model details
    DATASET_ID = "Dataset999_middleClick"
    CONFIG = "3d_fullres"

    bbox = [params.outer_point_one, params.outer_point_two]
    
    # replace this line when you change models
    new_seg = call_isac_model_predict(current_image, bbox, DATASET_ID, CONFIG, fold="0")

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

@app.post("/nnunet_predict")
async def nnunet_predict():
    """
    Runs nnU-Net prediction on the currently loaded image using the latest bounding box.
    This endpoint is now a wrapper around the reusable run_nnunet_segmentation function.
    """
    global current_image, segmentation_history
    if current_image is None:
        return {"status": "error", "message": "No image uploaded"}
    
    if not segmentation_history:
        return {"status": "error", "message": "No bounding box interaction found in history"}

    # Get the latest bounding box
    latest_bbox = segmentation_history[-1]['bbox']
    
    # IMPORTANT: Replace these with your actual model details
    DATASET_ID = "Dataset001_Tumors"
    CONFIG = "3d_fullres"

    # Run the segmentation
    seg_array = run_nnunet_segmentation(current_image, latest_bbox, DATASET_ID, CONFIG)

    if seg_array is None:
        return {"status": "error", "message": "nnU-Net prediction failed."}

    # Pack and compress for response
    compressed_result = pack_and_compress_segmentation(seg_array)
    
    print(f"Returning nnU-Net segmentation with {np.sum(seg_array)} voxels")
    return Response(
        content=compressed_result,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1527)