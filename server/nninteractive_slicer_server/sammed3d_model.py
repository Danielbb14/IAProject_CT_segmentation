"""
SAM-Med3D Model Predictor

This is for the SAM-Med3D checkpoint (12-layer teacher model).
SAM-Med3D is the full model that FastSAM3D was distilled from.
"""

import numpy as np
import torch
import torch.nn.functional as F
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
        # Resampling parameters
        self.original_spacing = (5.0, 0.703125, 0.703125)  # LITS default spacing
        self.target_spacing = 1.5  # SAM-Med3D training spacing (isotropic)
        self.zoom_factors = None

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
        """Store image and resample to isotropic spacing"""
        self.original_shape = image.shape

        # Resample to isotropic 1.5mm spacing
        print(f"SAM-Med3D: Original shape: {image.shape}, spacing: {self.original_spacing}")

        self.zoom_factors = [orig / self.target_spacing for orig in self.original_spacing]
        resampled = zoom(image, self.zoom_factors, order=1)  # bilinear interpolation

        self.current_image = resampled
        print(f"SAM-Med3D: Resampled to isotropic {self.target_spacing}mm: {resampled.shape}")
        print(f"SAM-Med3D: Zoom factors: {self.zoom_factors}")

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> np.ndarray:
        """Run prediction using cropping approach from FastSAM3D Slicer plugin"""
        if self.model is None or self.current_image is None:
            return self._mock_segmentation(point_coords)

        try:
            # Step 0: Adjust coordinates for resampled space
            adjusted_coords = point_coords.copy()
            adjusted_coords[:, 0] *= self.zoom_factors[2]  # x -> W
            adjusted_coords[:, 1] *= self.zoom_factors[1]  # y -> H
            adjusted_coords[:, 2] *= self.zoom_factors[0]  # z -> D

            print(f"\nOriginal coords [x,y,z]: {point_coords[0]}")
            print(f"Resampled coords [x,y,z]: {adjusted_coords[0]}")

            target_size = 128

            # Step 1: Pad image to at least 128x128x128
            pad_width = [(max(0, target_size - self.current_image.shape[i]) // 2,
                        max(0, target_size - self.current_image.shape[i]) // 2)
                        for i in range(3)]

            for i in range(3):
                if pad_width[i][0] + pad_width[i][1] + self.current_image.shape[i] < target_size:
                    pad_width[i] = (pad_width[i][0] + 1, pad_width[i][1])

            padded_image = np.pad(self.current_image, pad_width, 'constant')

            # Step 2: Convert coordinates from [x,y,z] to [D,H,W] and adjust for padding
            include_points = [[coords[2], coords[1], coords[0]] for coords in adjusted_coords]

            offsets = [pad_width[i][0] for i in range(3)]
            adjusted_points = [[coord + offset for coord, offset in zip(point, offsets)]
                            for point in include_points]

            print(f"Converted to [D,H,W]: {include_points[0]}")
            print(f"After padding: {adjusted_points[0]}")

            # Step 3: Find 128x128x128 box containing all points
            min_coords = []
            max_coords = []

            for i in range(3):
                coords_in_dim = [point[i] for point in adjusted_points]
                max_coord = max(coords_in_dim)
                min_coord = min(coords_in_dim)

                if max_coord - min_coord > target_size:
                    print(f"Points span {max_coord - min_coord} > {target_size} in dim {i}")
                    return np.zeros(self.original_shape, dtype=np.uint8)

                bound = padded_image.shape[i]
                crop = int((target_size - (max_coord - min_coord)) / 2)

                min_point = int(min_coord - min(min_coord, crop) - crop + min((bound - max_coord), crop))
                max_point = int(max_coord + min((bound - max_coord), crop) + crop - min(min_coord, crop))

                # Ensure exactly target_size
                if max_point - min_point != target_size:
                    if min_point > 0:
                        min_point -= 1
                    else:
                        max_point += 1

                min_coords.append(min_point)
                max_coords.append(max_point)

            # Step 4: Crop the region
            cropped_image = padded_image[min_coords[0]:max_coords[0],
                                        min_coords[1]:max_coords[1],
                                        min_coords[2]:max_coords[2]]

            # Step 5: Adjust coordinates to be relative to crop
            final_points = [[coord - offset for coord, offset in zip(point, min_coords)]
                        for point in adjusted_points]

            print(f"Crop box: D[{min_coords[0]}:{max_coords[0]}], H[{min_coords[1]}:{max_coords[1]}], W[{min_coords[2]}:{max_coords[2]}]")
            print(f"Coords in crop [D,H,W]: {final_points[0]}")

            # Step 6: Prepare tensors
            image_tensor = torch.from_numpy(cropped_image).float()
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

            points_tensor = torch.from_numpy(np.array(final_points)).float().unsqueeze(0).to(self.device)
            labels_tensor = torch.from_numpy(point_labels).long().unsqueeze(0).to(self.device)

            # Step 7: Run model (matching FastSAM3D plugin)
            prev_masks = torch.zeros_like(image_tensor).to(self.device)
            low_res_masks = F.interpolate(prev_masks.float(),
                                        size=(target_size//4, target_size//4, target_size//4))

            image_embeddings = self.model.image_encoder(image_tensor)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=[points_tensor, labels_tensor],
                boxes=None,
                masks=low_res_masks
            )

            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Step 8: Upsample to cropped size
            final_masks = F.interpolate(
                low_res_masks,
                size=cropped_image.shape,
                mode='trilinear',
                align_corners=False
            )

            mask_crop = torch.sigmoid(final_masks).detach().cpu().numpy().squeeze()
            mask_crop = (mask_crop > 0.5).astype(np.uint8)

            # Step 9: Place back in padded space
            mask_padded = np.zeros(padded_image.shape, dtype=np.uint8)
            mask_padded[min_coords[0]:max_coords[0],
                    min_coords[1]:max_coords[1],
                    min_coords[2]:max_coords[2]] = mask_crop

            # Step 10: Remove padding to get back to resampled size
            mask_resampled = mask_padded[pad_width[0][0]:mask_padded.shape[0] - pad_width[0][1],
                            pad_width[1][0]:mask_padded.shape[1] - pad_width[1][1],
                            pad_width[2][0]:mask_padded.shape[2] - pad_width[2][1]]

            # Step 11: Resample mask back to original spacing
            inverse_zoom = [1.0 / zf for zf in self.zoom_factors]
            mask_original = zoom(mask_resampled.astype(float), inverse_zoom, order=0)  # nearest neighbor for binary

            # Ensure exact original shape
            mask_original = mask_original[:self.original_shape[0],
                                         :self.original_shape[1],
                                         :self.original_shape[2]]
            mask_original = mask_original.astype(np.uint8)

            print(f"Resampled mask back to original shape: {mask_original.shape}")

            # Verify in original space
            nz = np.nonzero(mask_original)
            if len(nz[0]) > 0:
                center = [nz[0].mean(), nz[1].mean(), nz[2].mean()]
                clicked = [point_coords[0, 2], point_coords[0, 1], point_coords[0, 0]]
                dist = np.sqrt(sum((c - cl)**2 for c, cl in zip(center, clicked)))
                print(f"Mask center [D,H,W]: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
                print(f"Clicked [D,H,W]: [{clicked[0]:.1f}, {clicked[1]:.1f}, {clicked[2]:.1f}]")
                print(f"Distance: {dist:.1f}\n")

            return mask_original

        except Exception as e:
            print(f"SAM-Med3D prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_segmentation(point_coords)

    def _mock_segmentation(self, point_coords: np.ndarray) -> np.ndarray:
        """Fallback mock segmentation"""
        if self.current_image is None:
            return np.zeros((100, 100, 100), dtype=np.uint8)

        # Use original shape for mock
        mask = np.zeros(self.original_shape, dtype=np.uint8)

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