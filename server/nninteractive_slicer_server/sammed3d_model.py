"""
SAM-Med3D Model Predictor

This is for the SAM-Med3D checkpoint (12-layer teacher model).
SAM-Med3D is the full model that FastSAM3D was distilled from.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.ndimage import zoom

# SAM-Med3D imports (same as FastSAM3D)
try:
    from segment_anything.build_sam3D import sam_model_registry3D
    SAMMED3D_AVAILABLE = True
except ImportError:
    print("Warning: SAM-Med3D not installed")
    SAMMED3D_AVAILABLE = False


class FastSAM3DPredictor:
    """
    SAM-Med3D predictor (12-layer full model)
    Note: Keeping class name as FastSAM3DPredictor for compatibility with main.py
    """
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.current_image = None
        self.original_shape = None

    def load_model(self, checkpoint_path: str = "../checkpoints_data/sam_med3d_turbo.pth"):
        """Load SAM-Med3D model (12-layer full model)"""
        if self.model_loaded:
            return

        print(f"Loading SAM-Med3D on {self.device}...")

        if not SAMMED3D_AVAILABLE:
            print("SAM-Med3D not available")
            return

        try:
            # Use vit_b_ori (12-layer full model)
            print("Building SAM-Med3D (vit_b_ori, 12 layers)...")
            self.model = sam_model_registry3D["vit_b_ori"](checkpoint=None)

            # Load checkpoint
            print(f"Loading weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # SAM-Med3D checkpoints are nested under 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            # Fix normalization for grayscale medical images
            self.model.pixel_mean = torch.tensor([0.0], device=self.device).view(1, 1, 1, 1)
            self.model.pixel_std = torch.tensor([1.0], device=self.device).view(1, 1, 1, 1)

            self.model_loaded = True
            print("SAM-Med3D loaded successfully!")

        except Exception as e:
            print(f"Error loading SAM-Med3D: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = True

    def set_image(self, image: np.ndarray):
        """Store image for prediction"""
        self.current_image = image
        self.original_shape = image.shape
        print(f"SAM-Med3D: Image set with shape {image.shape}")

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> np.ndarray:
        """Run prediction with point prompts"""
        if self.model is None or self.current_image is None:
            return self._mock_segmentation(point_coords)

        try:
            # Resize to 128^3
            target_size = 128
            zoom_factors = [target_size / s for s in self.original_shape]
            resized_image = zoom(self.current_image, zoom_factors, order=1)

            # Prepare image tensor [C, D, H, W]
            image_tensor = torch.from_numpy(resized_image).float()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)  # [1, 128, 128, 128]
            image_tensor = image_tensor.to(self.device)

            # Scale coordinates
            # Scale coordinates - coords are [x, y, z] but shape is [z, y, x]
            scaled_coords = point_coords[:, [2, 1, 0]].copy()  # [x,y,z] -> [z,y,x]
            for i in range(3):
                scaled_coords[:, i] *= zoom_factors[i]

            print(f"DEBUG PREDICT: Input point_coords: {point_coords}")
            print(f"DEBUG PREDICT: Original shape: {self.original_shape}")
            print(f"DEBUG PREDICT: Zoom factors: {zoom_factors}")
            print(f"DEBUG PREDICT: Scaled coords: {scaled_coords}")

            # Prepare prompts
            point_coords_torch = torch.from_numpy(scaled_coords).float().unsqueeze(0)
            point_labels_torch = torch.from_numpy(point_labels).int().unsqueeze(0)
            point_coords_torch = point_coords_torch.to(self.device)
            point_labels_torch = point_labels_torch.to(self.device)

            # Build input
            batched_input = [{
                "image": image_tensor,  # [1, 128, 128, 128]
                "original_size": (target_size, target_size, target_size),
                "point_coords": point_coords_torch,
                "point_labels": point_labels_torch,
            }]

            # Run model
            with torch.no_grad():
                outputs = self.model(batched_input, multimask_output=False)

            print(f"SAM-Med3D: Model output keys: {outputs[0].keys()}")

            # Extract and resize mask
            masks = outputs[0]['masks']
            print(f"SAM-Med3D: Masks shape: {masks.shape}, dtype: {masks.dtype}")
            print(f"SAM-Med3D: Masks range: [{masks.min().item():.3f}, {masks.max().item():.3f}]")

            mask_128 = masks[0, 0].cpu().numpy()

            # Resize back to original shape
            zoom_back = [s / target_size for s in self.original_shape]
            mask = zoom(mask_128.astype(float), zoom_back, order=0)
            mask = (mask > 0.5).astype(np.uint8)

            print(f"SAM-Med3D: Final mask - shape: {mask.shape}, nonzero voxels: {np.count_nonzero(mask)}")

            return mask

        except Exception as e:
            print(f"SAM-Med3D prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_segmentation(point_coords)

    def _mock_segmentation(self, point_coords: np.ndarray) -> np.ndarray:
        """Fallback mock segmentation"""
        if self.current_image is None:
            return np.zeros((100, 100, 100), dtype=np.uint8)

        mask = np.zeros(self.current_image.shape, dtype=np.uint8)

        if len(point_coords) > 0:
            center = point_coords[0].astype(int)
            radius = 20

            for z in range(max(0, center[2]-radius), min(mask.shape[0], center[2]+radius)):
                for y in range(max(0, center[1]-radius), min(mask.shape[1], center[1]+radius)):
                    for x in range(max(0, center[0]-radius), min(mask.shape[2], center[0]+radius)):
                        dist = np.sqrt(
                            (x - center[0])**2 +
                            (y - center[1])**2 +
                            (z - center[2])**2
                        )
                        if dist <= radius:
                            mask[z, y, x] = 1

        return mask