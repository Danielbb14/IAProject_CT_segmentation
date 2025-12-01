import os
import tempfile
import subprocess
from typing import List, Union, Tuple
import numpy as np
import nibabel as nib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="ISAC nnU-Net Segmentation Server")


# -------------------------------
# Preprocessing: Extract Subvolume
# -------------------------------
def extract_subvolume_from_bbox(
    image: np.ndarray,
    bbox_coords: List[List[int]],
    margin: int = 30,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Extract a subvolume around the bounding box with added margin. 
    
    Args:
        image: Full CT image (3D numpy array)
        bbox_coords: [[x1, z1, y1], [x2, z2, y2]] bounding box corners
        margin: Additional padding around the bbox in voxels
        pad_value: Value to use for padding when subvolume extends beyond image bounds
    
    Returns:
        subvolume: Extracted subvolume around the bbox
        offset: (x_start, y_start, z_start) offset in original volume
        subvolume_shape: Shape of the extracted subvolume
    """
    p1, p2 = np. array(bbox_coords[0]), np.array(bbox_coords[1])
    
    # Get min/max coordinates (format: x, z, y)
    x_min, z_min, y_min = np.min([p1, p2], axis=0)
    x_max, z_max, y_max = np.max([p1, p2], axis=0)
    
    # Calculate center and size with margin
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2
    
    # Calculate bbox size with margin
    size_x = (x_max - x_min) + 2 * margin
    size_y = (y_max - y_min) + 2 * margin
    size_z = (z_max - z_min) + 2 * margin
    
    # Calculate subvolume bounds centered on bbox
    x_start = center_x - size_x // 2
    x_end = x_start + size_x
    y_start = center_y - size_y // 2
    y_end = y_start + size_y
    z_start = center_z - size_z // 2
    z_end = z_start + size_z
    
    # Create empty subvolume with padding
    subvolume = np.full((size_x, size_y, size_z), pad_value, dtype=image.dtype)
    
    # Calculate valid source ranges (clamped to image bounds)
    x_src_start = max(x_start, 0)
    x_src_end = min(x_end, image.shape[0])
    y_src_start = max(y_start, 0)
    y_src_end = min(y_end, image.shape[1])
    z_src_start = max(z_start, 0)
    z_src_end = min(z_end, image. shape[2])
    
    # Calculate destination indices in subvolume
    x_dst_start = x_src_start - x_start
    x_dst_end = x_dst_start + (x_src_end - x_src_start)
    y_dst_start = y_src_start - y_start
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    z_dst_start = z_src_start - z_start
    z_dst_end = z_dst_start + (z_src_end - z_src_start)
    
    # Copy valid region from image to subvolume
    subvolume[x_dst_start:x_dst_end, y_dst_start:y_dst_end, z_dst_start:z_dst_end] = \
        image[x_src_start:x_src_end, y_src_start:y_src_end, z_src_start:z_src_end]
    
    # Store offset for postprocessing
    offset = (x_start, y_start, z_start)
    
    print(f"Extracted subvolume: shape={subvolume.shape}, offset={offset}")
    print(f"  Original bbox: x=[{x_min},{x_max}], y=[{y_min},{y_max}], z=[{z_min},{z_max}]")
    print(f"  Subvolume bounds: x=[{x_start},{x_end}], y=[{y_start},{y_end}], z=[{z_start},{z_end}]")
    
    return subvolume, offset, subvolume.shape


# -------------------------------
# Postprocessing: Place Back in Full Volume
# -------------------------------
def place_prediction_in_full_volume(
    prediction: np.ndarray,
    original_shape: Tuple[int, int, int],
    offset: Tuple[int, int, int]
) -> np.ndarray:
    """
    Place the subvolume prediction back into a full-size volume.
    
    Args:
        prediction: Segmentation prediction on the subvolume
        original_shape: Shape of the original full CT image
        offset: (x_start, y_start, z_start) offset where subvolume was extracted
    
    Returns:
        full_volume_prediction: Prediction mask in full volume space
    """
    # Create empty full-size volume
    full_prediction = np.zeros(original_shape, dtype=prediction.dtype)
    
    x_start, y_start, z_start = offset
    x_end = x_start + prediction.shape[0]
    y_end = y_start + prediction.shape[1]
    z_end = z_start + prediction.shape[2]
    
    # Calculate valid ranges (clamped to original volume bounds)
    x_dst_start = max(x_start, 0)
    x_dst_end = min(x_end, original_shape[0])
    y_dst_start = max(y_start, 0)
    y_dst_end = min(y_end, original_shape[1])
    z_dst_start = max(z_start, 0)
    z_dst_end = min(z_end, original_shape[2])
    
    # Calculate source indices in prediction
    x_src_start = x_dst_start - x_start
    x_src_end = x_src_start + (x_dst_end - x_dst_start)
    y_src_start = y_dst_start - y_start
    y_src_end = y_src_start + (y_dst_end - y_dst_start)
    z_src_start = z_dst_start - z_start
    z_src_end = z_src_start + (z_dst_end - z_dst_start)
    
    # Place prediction in full volume
    full_prediction[x_dst_start:x_dst_end, y_dst_start:y_dst_end, z_dst_start:z_dst_end] = \
        prediction[x_src_start:x_src_end, y_src_start:y_src_end, z_src_start:z_src_end]
    
    print(f"Placed prediction back: {np.sum(prediction)} -> {np.sum(full_prediction)} voxels")
    
    return full_prediction


# -------------------------------
# nnU-Net segmentation function
# -------------------------------
def run_nnunet_segmentation(
    image: np.ndarray,
    bbox_coords: List[List[int]],
    dataset_id: str,
    config: str,
    fold: str
) -> Union[np.ndarray, None]:
    """
    Runs nnU-Net prediction on a given image and bounding box.
    """

    import json

    p1, p2 = np. array(bbox_coords[0]), np.array(bbox_coords[1])

    # Determine the bounding box slices (assuming z, y, x order)
    x_min, z_min, y_min = np.min([p1, p2], axis=0)
    x_max, z_max, y_max = np.max([p1, p2], axis=0)

    # Create the bounding box channel
    bbox_channel = np. zeros(image.shape, dtype=np.uint8)
    bbox_channel[x_min:x_max+1, z_min:z_max+1, y_min:y_max+1] = 1

    # Create temporary directories for nnU-Net input/output
    with tempfile. TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os. path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        try:
            # 1. Save the current image and bbox channel as NIfTI files
            image_path_0000 = os.path.join(input_dir, "image_0000.nii.gz")
            image_path_0001 = os.path.join(input_dir, "image_0001.nii.gz")

            affine = np.eye(4)

            nib.save(nib. Nifti1Image(image, affine), image_path_0000)
            nib.save(nib.Nifti1Image(bbox_channel, affine), image_path_0001)

            # Set nnU-Net environment variables
            # Change these paths as needed for your model setup
            os.environ["nnUNet_results"] = "/home/moriarty_d/projects/nnunet-bbox/nnunet_results"
            # os.environ["nnUNet_preprocessed"] = "/home/moriarty_d/projects/nnunet-oneclick/nnunet_preprocessed"
            # os.environ["nnUNet_raw"] = "/home/moriarty_d/projects/nnunet-oneclick/nnunet_raw"

            # 2.  Construct nnU-Net command
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

            # 3. Load output segmentation
            output_path = os.path.join(output_dir, "image. nii.gz")
            if not os.path.exists(output_path):
                raise FileNotFoundError("nnU-Net did not produce an output file.")

            seg_img = nib.load(output_path)
            seg_array = seg_img.get_fdata(). astype(np.uint8)
            return seg_array

        except subprocess.CalledProcessError as e:
            print(f"nnU-Net command failed (exit {e.returncode}): {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error during nnU-Net prediction: {e}")
            return None


# -------------------------------
# FastAPI endpoint
# -------------------------------
class IsacModelPredictRequest(BaseModel):
    image: List[List[List[float]]]  # JSON-serializable ndarray
    bbox_coords: List[List[int]]
    dataset_id: str
    config: str
    fold: str
    margin: int = 30  # Optional: margin around bbox for subvolume extraction


@app.post("/IsacModelPredict")
async def isac_model_predict(req: IsacModelPredictRequest):
    # Convert image to numpy array
    np_image = np.array(req.image, dtype=np.float32)
    original_shape = np_image.shape
    
    print(f"Received image shape: {original_shape}")
    print(f"Bounding box: {req.bbox_coords}")
    
    # PREPROCESSING: Extract subvolume around bounding box
    subvolume, offset, subvolume_shape = extract_subvolume_from_bbox(
        np_image,
        req.bbox_coords,
        margin=req.margin,
        pad_value=0.0
    )
    
    # Update bbox coords to be relative to the subvolume
    # Original bbox coords are in full image space, need to subtract offset
    p1, p2 = np. array(req.bbox_coords[0]), np.array(req. bbox_coords[1])
    x_offset, y_offset, z_offset = offset
    
    # Adjust coordinates: subtract offset (format: x, z, y)
    relative_bbox = [
        [p1[0] - x_offset, p1[1] - z_offset, p1[2] - y_offset],
        [p2[0] - x_offset, p2[1] - z_offset, p2[2] - y_offset]
    ]
    
    # Clamp to subvolume bounds
    relative_bbox[0] = np.clip(relative_bbox[0], 0, [subvolume_shape[0]-1, subvolume_shape[2]-1, subvolume_shape[1]-1]). tolist()
    relative_bbox[1] = np.clip(relative_bbox[1], 0, [subvolume_shape[0]-1, subvolume_shape[2]-1, subvolume_shape[1]-1]).tolist()
    
    print(f"Relative bbox in subvolume: {relative_bbox}")

    # Run nnU-Net segmentation on subvolume
    seg_result = run_nnunet_segmentation(
        subvolume,  # Use subvolume instead of full image
        relative_bbox,  # Use relative coordinates
        req.dataset_id,
        req.config,
        req.fold
    )

    if seg_result is None:
        return {"prediction": None, "status": "error"}
    
    # POSTPROCESSING: Place prediction back in full volume
    full_volume_prediction = place_prediction_in_full_volume(
        seg_result,
        original_shape,
        offset
    )

    # Convert result back to list for JSON serialization
    return {"prediction": full_volume_prediction. tolist(), "status": "success"}