import os
import tempfile
import subprocess
from typing import List, Union
import numpy as np
import nibabel as nib
from fastapi import FastAPI
from pydantic import BaseModel






app = FastAPI(title="ISAC nnU-Net Segmentation Server")


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

    p1, p2 = np.array(bbox_coords[0]), np.array(bbox_coords[1])

    # Determine the bounding box slices (assuming z, y, x order)
    z_min, y_min, x_min = np.min([p1, p2], axis=0)
    z_max, y_max, x_max = np.max([p1, p2], axis=0)

    # Create the bounding box channel
    bbox_channel = np.zeros(image.shape, dtype=np.uint8)
    bbox_channel[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 1

    # Create temporary directories for nnU-Net input/output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        try:
            # 1. Save the current image and bbox channel as NIfTI files
            image_path_0000 = os.path.join(input_dir, "image_0000.nii.gz")
            image_path_0001 = os.path.join(input_dir, "image_0001.nii.gz")

            affine = np.eye(4)

            nib.save(nib.Nifti1Image(image, affine), image_path_0000)
            nib.save(nib.Nifti1Image(bbox_channel, affine), image_path_0001)

            # Set nnU-Net environment variables
            os.environ["nnUNet_results"] = "/home/moriarty_d/projects/nnunet-oneclick/nnunet_results"
            os.environ["nnUNet_preprocessed"] = "/home/moriarty_d/projects/nnunet-oneclick/nnunet_preprocessed"
            os.environ["nnUNet_raw"] = "/home/moriarty_d/projects/nnunet-oneclick/nnunet_raw"

            # 2. Construct nnU-Net command
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
            output_path = os.path.join(output_dir, "image.nii.gz")
            if not os.path.exists(output_path):
                raise FileNotFoundError("nnU-Net did not produce an output file.")

            seg_img = nib.load(output_path)
            seg_array = seg_img.get_fdata().astype(np.uint8)
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


@app.post("/IsacModelPredict")
async def isac_model_predict(req: IsacModelPredictRequest):
    # Convert image to numpy array
    np_image = np.array(req.image, dtype=np.float32)

    # Run nnU-Net segmentation
    seg_result = run_nnunet_segmentation(
        np_image,
        req.bbox_coords,
        req.dataset_id,
        req.config,
        req.fold
    )

    if seg_result is None:
        return {"prediction": None, "status": "error"}

    # Convert result back to list for JSON serialization
    return {"prediction": seg_result.tolist(), "status": "success"}
