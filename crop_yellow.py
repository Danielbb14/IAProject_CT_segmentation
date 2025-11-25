import nibabel as nib
import numpy as np
from scipy.ndimage import label
from pathlib import Path
import argparse

def crop_to_cube(volume, indices, cube_size, pad_value=0):
    x_min, x_max = indices[:, 0].min(), indices[:, 0].max()
    y_min, y_max = indices[:, 1].min(), indices[:, 1].max()
    z_min, z_max = indices[:, 2].min(), indices[:, 2].max()

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2

    sx, sy, sz = cube_size
    cx0, cx1 = center_x - sx // 2, center_x - sx // 2 + sx
    cy0, cy1 = center_y - sy // 2, center_y - sy // 2 + sy
    cz0, cz1 = center_z - sz // 2, center_z - sz // 2 + sz

    # prepare empty cube
    cube = np.full((sx, sy, sz), pad_value, dtype=volume.dtype)

    # compute valid ranges
    x0_src, x1_src = max(cx0, 0), min(cx1, volume.shape[0])
    y0_src, y1_src = max(cy0, 0), min(cy1, volume.shape[1])
    z0_src, z1_src = max(cz0, 0), min(cz1, volume.shape[2])

    # destination indices
    x0_dst, x1_dst = x0_src - cx0, x0_src - cx0 + (x1_src - x0_src)
    y0_dst, y1_dst = y0_src - cy0, y0_src - cy0 + (y1_src - y0_src)
    z0_dst, z1_dst = z0_src - cz0, z0_src - cz0 + (z1_src - z0_src)

    cube[x0_dst:x1_dst, y0_dst:y1_dst, z0_dst:z1_dst] = volume[x0_src:x1_src, y0_src:y1_src, z0_src:z1_src]

    return cube

def crop_yellow_regions_with_seg(ct_path, seg_path, cube_mm, output_dir=None):
    ct_nii = nib.load(ct_path)
    seg_nii = nib.load(seg_path)

    ct = ct_nii.get_fdata()
    seg = seg_nii.get_fdata()

    # Only keep yellow regions (label=2) for cropping centers
    mask = (seg == 2).astype(np.uint8)
    labeled_mask, num_components = label(mask)

    voxel_spacing = ct_nii.header.get_zooms()[:3]
    cube_size = tuple(int(np.ceil(cube_mm / vs)) for vs in voxel_spacing)

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    for comp_id in range(1, num_components + 1):
        indices = np.argwhere(labeled_mask == comp_id)
        if indices.size == 0:
            continue

        # Crop CT
        ct_cube = crop_to_cube(ct, indices, cube_size, pad_value=0)
        ct_out_file = output_dir / f"{Path(ct_path).stem}_yellow{comp_id}.nii.gz"
        nib.save(nib.Nifti1Image(ct_cube, affine=ct_nii.affine), ct_out_file)

        # Crop segmentation (same indices, same cube size)
        seg_cube = crop_to_cube(seg, indices, cube_size, pad_value=0)
        seg_out_file = output_dir / f"{Path(seg_path).stem}_yellow{comp_id}_seg.nii.gz"
        nib.save(nib.Nifti1Image(seg_cube, affine=seg_nii.affine), seg_out_file)

        print(f"Saved CT: {ct_out_file.name}  Segmentation: {seg_out_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ct_path")
    parser.add_argument("seg_path")
    parser.add_argument("--cube_mm", type=float, default=10, help="Cube size in mm")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save cubes")
    args = parser.parse_args()

    crop_yellow_regions_with_seg(args.ct_path, args.seg_path, args.cube_mm, args.output_dir)
