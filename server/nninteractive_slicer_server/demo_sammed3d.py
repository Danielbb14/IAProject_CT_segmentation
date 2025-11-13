"""
Quick Demo: SAM-Med3D Integration
Shows the model loading and making a prediction on volume-0.nii
"""

import numpy as np
import nibabel as nib
import sys
sys.path.insert(0, '/domus/h1/junming/private/SAM-Med3D')

from sammed3d_model import FastSAM3DPredictor

print("="*60)
print("SAM-Med3D Integration Demo")
print("="*60)
print()

# Load the model
print("1. Loading SAM-Med3D model...")
predictor = FastSAM3DPredictor()
predictor.load_model("../../checkpoints_data/sam_med3d_turbo.pth")
print("   ✓ Model loaded successfully!")
print()

# Load test volume
print("2. Loading test volume (volume-0.nii)...")
volume_path = "/domus/h1/junming/volume-0.nii"
nii = nib.load(volume_path)
volume = nii.get_fdata()
print(f"   ✓ Volume shape: {volume.shape}")
print(f"   ✓ Value range: [{volume.min():.1f}, {volume.max():.1f}]")
print()

# Set image
print("3. Setting image in predictor...")
predictor.set_image(volume)
print("   ✓ Image set!")
print()

# Create a point prompt (center of volume)
center = [volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2]
print(f"4. Creating point prompt at center: {center}")
point_coords = np.array([center], dtype=np.float32)
point_labels = np.array([1], dtype=np.int32)  # 1 = positive
print()

# Run prediction
print("5. Running SAM-Med3D prediction...")
mask = predictor.predict(point_coords, point_labels)
print(f"   ✓ Prediction complete!")
print(f"   ✓ Mask shape: {mask.shape}")
print(f"   ✓ Segmented voxels: {np.sum(mask)}")
print()

print("="*60)
print("Demo Complete! SAM-Med3D is working! 🎉")
print("="*60)
