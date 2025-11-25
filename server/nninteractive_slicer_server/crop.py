import nibabel as nib
import numpy as np
from scipy.ndimage import label
from pathlib import Path
import re
import argparse


def crop_to_cube(target, comp_indices, target_shape, pad_value=None):
    """
    Crop around a component to a fixed cube shape.
    If the crop exceeds the image boundary, pad with `pad_value`.
    """
    # bounding box center
    xMin, xMax = comp_indices[:, 0].min(), comp_indices[:, 0].max()
    yMin, yMax = comp_indices[:, 1].min(), comp_indices[:, 1].max()
    zMIn, zMax = comp_indices[:, 2].min(), comp_indices[:, 2].max()

    centreOfX, centreOfY, centreOfZ = (
        (xMin + xMax) // 2,
        (yMin + yMax) // 2,
        (zMIn + zMax) // 2,
    )
    desiredVoxelsX, desiredVoxelsY, desiredVoxelsZ = target_shape
    currentVoxelsX, currentVoxelsY, currentVoxelsZ = target.shape

    # desired crop ranges
    x0, x1 = (
        centreOfX - desiredVoxelsX // 2,
        centreOfX - desiredVoxelsX // 2 + desiredVoxelsX,
    )
    y0, y1 = (
        centreOfY - desiredVoxelsY // 2,
        centreOfY - desiredVoxelsY // 2 + desiredVoxelsY,
    )
    z0, z1 = (
        centreOfZ - desiredVoxelsZ // 2,
        centreOfZ - desiredVoxelsZ // 2 + desiredVoxelsZ,
    )

    if pad_value is None:
        cropped_volume = np.zeros(
            (desiredVoxelsX, desiredVoxelsY, desiredVoxelsZ),
            dtype=target.dtype
        )
    else:
        cropped_volume = np.full(
            (desiredVoxelsX, desiredVoxelsY, desiredVoxelsZ),
            pad_value,
            dtype=target.dtype,
        )

    # compute valid ranges (intersection with image bounds)
    x0_src, x1_src = max(0, x0), min(currentVoxelsX, x1)
    y0_src, y1_src = max(0, y0), min(currentVoxelsY, y1)
    z0_src, z1_src = max(0, z0), min(currentVoxelsZ, z1)

    # corresponding destination ranges in crop
    x0_dst, x1_dst = x0_src - x0, x0_src - x0 + (x1_src - x0_src)
    y0_dst, y1_dst = y0_src - y0, y0_src - y0 + (y1_src - y0_src)
    z0_dst, z1_dst = z0_src - z0, z0_src - z0 + (z1_src - z0_src)

    # copy image and mask into padded crop
    cropped_volume[x0_dst:x1_dst, y0_dst:y1_dst, z0_dst:z1_dst] = target[
        x0_src:x1_src, y0_src:y1_src, z0_src:z1_src
    ]

    return cropped_volume


def crop_lesions_to_cubes(
    target_to_crop,
    source_mask,
    target_side_mm,
    output_dir=None,
    pad_value=None,
    append_to_path="",
    is_binary=False,
):
    image_nii = nib.load(str(target_to_crop))
    mask_nii = nib.load(str(source_mask))

    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()

    # --- Handle 4D images (x, y, z, 1) ---
    if image.ndim == 4 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)

    voxelSpacings = image_nii.header.get_zooms()[:3]
    target_shape = tuple(
        int(np.ceil(target_side_mm / spacing)) for spacing in voxelSpacings
    )

    crops = []
    labeled_mask, num_components = label(mask)

    for comp_id in range(1, num_components + 1):
        comp_indices = np.argwhere(labeled_mask == comp_id)
        if comp_indices.size == 0:
            continue

        base = re.sub(r"(_\d{4})(?=\.nii(\.gz)?$)", "", str(target_to_crop))

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            img_file = output_dir / re.sub(
                r"(?=\.nii(\.gz)?$)",
                f"{append_to_path}",
                f"{Path(base).stem}_lesion{comp_id}.nii.gz",
            )

            # --- Skip if output already exists ---
            if img_file.exists():
                print(f"Skipping {img_file.name} (already exists)")
                continue

        if is_binary:
            labeled_target, n_target = label(image.astype(np.uint8))
            overlapping_labels = np.unique(
                labeled_target[(labeled_mask == comp_id)]
            )
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            target_overlap_mask = np.isin(labeled_target, overlapping_labels)

            img_crop = crop_to_cube(
                target_overlap_mask.astype(np.uint8),
                comp_indices,
                target_shape,
                pad_value=0,
            )
        else:
            img_crop = crop_to_cube(
                image,
                comp_indices,
                target_shape,
                pad_value=pad_value
            )

        if output_dir:
            img_out = nib.Nifti1Image(img_crop, affine=image_nii.affine)
            nib.save(img_out, img_file)
            print(f"Saved {img_file.name}")
        else:
            crops.append((img_crop))

    return crops if not output_dir else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop an image to a cube.")
    parser.add_argument(
        "target_image",
        type=str,
        help="Directory containing NIfTI files (.nii or .nii.gz)",
    )
    parser.add_argument(
        "source_image",
        type=str,
        help="Directory containing NIfTI files (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--target_side_mm",
        type=float,
        default=5,
        help="Size to crop",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--pad_value",
        type=int,
        default=None,
        help="Value to pad extra padding",
    )
    parser.add_argument(
        "--append_to_path",
        type=str,
        default="",
        help="Ending to add to file",
    )
    parser.add_argument(
        "--is_binary",
        type=bool,
        default=True,
        help="Bool option whether to treat d_min as a relative distance to the lesion size",
    )

    args = parser.parse_args()

    crop_lesions_to_cubes(
        args.target_image,
        args.source_image,
        target_side_mm=args.target_side_mm,
        output_dir=args.output_dir,
        pad_value=args.pad_value,
        append_to_path="",
        is_binary=False,
    )
