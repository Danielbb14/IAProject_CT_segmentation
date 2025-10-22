import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import gzip

app = FastAPI()

# Global storage for the uploaded image and segment
current_image = None
current_segment = None

class BBoxParams(BaseModel):
    outer_point_one: list[int]  # [x, y, z] - already reversed by plugin
    outer_point_two: list[int]  # [x, y, z] - already reversed by plugin
    positive_click: bool = True

def create_mock_segmentation(image_shape, bbox_coords):
    """
    Create a mock segmentation - small ball/lesion in the bounding box center
    Note: bbox_coords are already in reversed order from the plugin
    """
    # Create empty segmentation
    seg = np.zeros(image_shape, dtype=np.uint8)
    
    # Calculate bounding box center (coordinates already reversed by plugin)
    p1, p2 = np.array(bbox_coords[0]), np.array(bbox_coords[1])
    center = ((p1 + p2) / 2).astype(int)
    
    # Ensure center is within image bounds
    center = np.clip(center, 0, np.array(image_shape) - 1)
    
    # Create a small sphere around the center (radius ~10 voxels)
    radius = 10
    x, y, z = np.ogrid[:image_shape[0], :image_shape[1], :image_shape[2]]
    
    # Distance from center
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    # Create sphere
    seg[dist <= radius] = 1
    
    return seg

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
    return {"status": "ok"}  # Use "ok" not "success" to match original

@app.post("/upload_segment")
async def upload_segment(file: UploadFile = File(...)):
    """
    Upload and store the segmentation (plugin sends this automatically)
    """
    global current_segment, current_image
    
    # Check if image is uploaded first (same as original server)
    if current_image is None:
        return {"status": "error", "message": "No image uploaded"}
    
    # Read, decompress and load the numpy array (same as original)
    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)
    current_segment = np.load(io.BytesIO(decompressed))
    
    print(f"Segment uploaded with shape: {current_segment.shape}")
    return {"status": "ok"}

@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    """
    Take bounding box coordinates and return mock segmentation
    Plugin sends coordinates in REVERSED order, so we use them as-is
    """
    global current_image
    
    # Check if image is uploaded (same error format as original)
    if current_image is None:
        return {"status": "error", "message": "No image uploaded"}
    
    print(f"Received bbox (already reversed): {params.outer_point_one} to {params.outer_point_two}")
    
    # Create mock segmentation (coordinates already reversed by plugin)
    mock_seg = create_mock_segmentation(
        current_image.shape, 
        [params.outer_point_one, params.outer_point_two]
    )
    
    # Pack and compress the result
    compressed_result = pack_and_compress_segmentation(mock_seg)
    
    print(f"Returning segmentation with {np.sum(mock_seg)} positive voxels")
    
    return Response(
        content=compressed_result,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )

@app.get("/status")
async def get_status():
    """
    Check server status and current image info
    """
    global current_image, current_segment
    
    return {
        "status": "ready", 
        "image_loaded": current_image is not None,
        "segment_loaded": current_segment is not None,
        "image_shape": list(current_image.shape) if current_image is not None else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1527)  # Use same port as original