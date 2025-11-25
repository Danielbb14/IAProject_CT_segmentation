import nibabel as nib
import numpy as np

img = nib.load("/Users/dbb14/Desktop/Uppsala University/period 5/IA_project/media/nas/01_Datasets/CT/LITS/Training Batch 1/segmentation-0.nii")
data = img.get_fdata()

print("Unique labels:", np.unique(data))
