import unittest
import numpy as np
import nibabel as nib
import os
import sys

# Add the server directory to the Python path to allow importing main3
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from nninteractive_slicer_server.main3 import run_nnunet_segmentation

class TestIntegrationMain3(unittest.TestCase):

    def test_end_to_end_nnunet_segmentation(self):
        """
        Tests the full end-to-end flow of the run_nnunet_segmentation function. 
        
        This is an INTEGRATION TEST, not a unit test. It requires:
        1. A fully configured nnU-Net v2 environment.
        2. The necessary environment variables (nnUNet_results, etc.) to be set.
        3. The test data file to be present at TEST_IMAGE_PATH.
        """
        # --- Test Configuration ---
        # The path to the test image, relative to the 'server' directory
        TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_data", "volume-1_0000.nii.gz")
        
        # The bounding box coordinates provided by the user
        BBOX_COORDS = [[256, 87, 55], [256, 189, 67]]
        
        # The model parameters (must match the model being tested)
        DATASET_ID = "Dataset999_middleClick"
        CONFIG = "3d_fullres"
        FOLD = "0"

        # --- Test Execution ---
        
        # 1. Check if the test image exists
        self.assertTrue(os.path.exists(TEST_IMAGE_PATH), f"Test image not found at {TEST_IMAGE_PATH}")

        # 2. Load the image data
        try:
            image_nifti = nib.load(TEST_IMAGE_PATH)
            image_data = image_nifti.get_fdata()
        except Exception as e:
            self.fail(f"Failed to load the test image file: {e}")

        # 3. Run the actual segmentation function
        print("\n--- Running Integration Test: Calling run_nnunet_segmentation ---")
        print(f"Image shape: {image_data.shape}")
        print(f"Bounding Box: {BBOX_COORDS}")
        
        result_array = run_nnunet_segmentation(
            image=image_data,
            bbox_coords=BBOX_COORDS,
            dataset_id=DATASET_ID,
            config=CONFIG,
            fold=FOLD
        )
        
        print("--- Call to run_nnunet_segmentation finished ---")

        # --- Verification ---
        
        # 4. Ensure the model successfully produced a prediction
        self.assertIsNotNone(result_array, "The segmentation function returned None, indicating a failure in the prediction process.")
        
        # 5. Ensure the model output conforms to the required format
        self.assertIsInstance(result_array, np.ndarray, "Output is not a NumPy array.")
        self.assertEqual(result_array.dtype, np.uint8, f"Output array is of type {result_array.dtype}, but should be np.uint8.")
        self.assertEqual(image_data.shape, result_array.shape, "The shape of the output segmentation does not match the input image.")

        print("--- Integration Test Passed ---")
        print(f"Output segmentation generated with shape: {result_array.shape} and type: {result_array.dtype}")
        print(f"Total segmented voxels: {np.sum(result_array)}")


if __name__ == '__main__':
    # This allows the test to be run directly from the command line
    unittest.main()
