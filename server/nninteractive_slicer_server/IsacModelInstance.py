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
    cube_size: int = 20,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, Tuple[int, int, int], List[List[int]]]:
    """
    Extract a FIXED-SIZE subvolume centered on the bounding box.
    """
    p1, p2 = np. array(bbox_coords[0]), np.array(bbox_coords[1])
    
    # Get bbox extents (format: x, z, y)
    x_min, z_min, y_min = np.min([p1, p2], axis=0)
    x_max, z_max, y_max = np.max([p1, p2], axis=0)
    
    # Calculate center of bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2
    
    # FIXED subvolume size
    size_x = cube_size
    size_y = cube_size
    size_z = cube_size
    
    # Calculate subvolume bounds centered on bbox center
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
    y_src_end = min(y_end, image. shape[1])
    z_src_start = max(z_start, 0)
    z_src_end = min(z_end, image.shape[2])
    
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
    
    # Calculate bbox coordinates relative to the subvolume
    relative_x_min = x_min - x_start
    relative_x_max = x_max - x_start
    relative_z_min = z_min - z_start
    relative_z_max = z_max - z_start
    relative_y_min = y_min - y_start
    relative_y_max = y_max - y_start
    
    # Clamp to subvolume bounds
    relative_x_min = max(0, min(relative_x_min, size_x - 1))
    relative_x_max = max(0, min(relative_x_max, size_x - 1))
    relative_z_min = max(0, min(relative_z_min, size_z - 1))
    relative_z_max = max(0, min(relative_z_max, size_z - 1))
    relative_y_min = max(0, min(relative_y_min, size_y - 1))
    relative_y_max = max(0, min(relative_y_max, size_y - 1))
    
    relative_bbox = [
        [relative_x_min, relative_z_min, relative_y_min],
        [relative_x_max, relative_z_max, relative_y_max]
    ]
    
    print(f"[PREPROCESS] Original image shape: {image.shape}")
    print(f"[PREPROCESS] FIXED subvolume size: {cube_size}x{cube_size}x{cube_size} voxels")
    print(f"[PREPROCESS] Extracted subvolume shape: {subvolume.shape}")
    print(f"[PREPROCESS] Offset: {offset}")
    print(f"[PREPROCESS] Original bbox: {bbox_coords}")
    print(f"[PREPROCESS] Relative bbox: {relative_bbox}")
    
    return subvolume, offset, relative_bbox


# -------------------------------
# Postprocessing: Place Back in Full Volume
# -------------------------------
def place_prediction_in_full_volume(
    prediction: np. ndarray,
    original_shape: Tuple[int, int, int],
    offset: Tuple[int, int, int]
) -> np.ndarray:
    """
    Place the subvolume prediction back into a full-size volume. 
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
    
    print(f"[POSTPROCESS] Prediction shape: {prediction.shape}")
    print(f"[POSTPROCESS] Prediction voxels: {np.sum(prediction)}")
    print(f"[POSTPROCESS] Full volume shape: {full_prediction.shape}")
    print(f"[POSTPROCESS] Full volume voxels: {np.sum(full_prediction)}")
    
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
) -> Union[np. ndarray, None]:
    """
    Runs nnU-Net prediction on a given image and bounding box.
    """

    print(f"[NNUNET] Image shape: {image. shape}")
    print(f"[NNUNET] Bbox coords: {bbox_coords}")

    p1, p2 = np. array(bbox_coords[0]), np.array(bbox_coords[1])

    # Determine the bounding box slices
    x_min, z_min, y_min = np. min([p1, p2], axis=0)
    x_max, z_max, y_max = np.max([p1, p2], axis=0)

    print(f"[NNUNET] Bbox bounds: x=[{x_min},{x_max}], y=[{y_min},{y_max}], z=[{z_min},{z_max}]")

    # Create the bounding box channel
    bbox_channel = np.zeros(image.shape, dtype=np.uint8)
    bbox_channel[x_min:x_max+1, z_min:z_max+1, y_min:y_max+1] = 1

    print(f"[NNUNET] Bbox channel sum: {bbox_channel.sum()} voxels")

    # Create temporary directories for nnU-Net input/output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os. path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        try:
            # 1. Save the current image and bbox channel as NIfTI files
            image_path_0000 = os.path. join(input_dir, "image_0000.nii.gz")
            image_path_0001 = os.path.join(input_dir, "image_0001.nii.gz")

            affine = np.eye(4)

            nib.save(nib.Nifti1Image(image, affine), image_path_0000)
            nib.save(nib. Nifti1Image(bbox_channel, affine), image_path_0001)

            # Set nnU-Net environment variables
            os.environ["nnUNet_results"] = "/home/moriarty_d/projects/nnunet-bbox/nnunet_results"

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

            print(f"[NNUNET] Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            print("[NNUNET] nnU-Net completed successfully")

            # 3. Load output segmentation - TRY MULTIPLE POSSIBLE FILENAMES
            # nnU-Net might name it "image.nii.gz" directly
            possible_output_paths = [
                os.path. join(output_dir, "image.nii.gz"),  # Direct output
                os.path.join(output_dir, "image_0000.nii.gz"),  # Sometimes keeps input name
            ]
            
            output_path = None
            for path in possible_output_paths:
                if os.path.exists(path):
                    output_path = path
                    print(f"[NNUNET] Found output at: {output_path}")
                    break
            
            if output_path is None:
                print(f"[NNUNET ERROR] No output found. Available files: {os.listdir(output_dir)}")
                raise FileNotFoundError("nnU-Net did not produce an output file.")

            seg_img = nib.load(output_path)
            seg_array = seg_img.get_fdata().astype(np.uint8)
            
            print(f"[NNUNET] Segmentation shape: {seg_array. shape}, sum: {seg_array.sum()}")
            
            return seg_array

        except subprocess.CalledProcessError as e:
            print(f"[NNUNET ERROR] Command failed (exit {e.returncode}): {e.stderr}")
            return None
        except Exception as e:
            print(f"[NNUNET ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None


# -------------------------------
# FastAPI endpoint
# -------------------------------
class IsacModelPredictRequest(BaseModel):
    image: List[List[List[float]]]
    bbox_coords: List[List[int]]
    dataset_id: str
    config: str
    fold: str
    cube_size: int = 20


@app.post("/IsacModelPredict")
async def isac_model_predict(req: IsacModelPredictRequest):
    print("\n" + "="*80)
    print("[ENDPOINT] IsacModelPredict called")
    print("="*80)
    
    # Convert image to numpy array
    np_image = np.array(req.image, dtype=np.float32)
    original_shape = np_image.shape
    
    print(f"[ENDPOINT] Original image shape: {original_shape}")
    print(f"[ENDPOINT] Original bbox: {req.bbox_coords}")
    print(f"[ENDPOINT] Cube size: {req.cube_size} voxels")
    
    # PREPROCESSING: Extract FIXED-SIZE subvolume around bounding box
    subvolume, offset, relative_bbox = extract_subvolume_from_bbox(
        np_image,
        req.bbox_coords,
        cube_size=req.cube_size,
        pad_value=0.0
    )

    # Run nnU-Net segmentation on subvolume with relative bbox coords
    seg_result = run_nnunet_segmentation(
        subvolume,
        relative_bbox,
        req.dataset_id,
        req.config,
        req.fold
    )

    if seg_result is None:
        print("[ENDPOINT] Segmentation failed!")
        return {"prediction": None, "status": "error"}
    
    # POSTPROCESSING: Place prediction back in full volume
    full_volume_prediction = place_prediction_in_full_volume(
        seg_result,
        original_shape,
        offset
    )

    print(f"[ENDPOINT] Success! Returning {np.sum(full_volume_prediction)} voxels")
    print("="*80 + "\n")
    
    # Convert result back to list for JSON serialization
    return {"prediction": full_volume_prediction. tolist(), "status":"success"}