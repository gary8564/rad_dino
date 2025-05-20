# tests/test_vindrcxr_dataset.py

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# adjust this import to point at your dataset implementation
from rad_dino.data.VinDrCXR.data import VinDrCXR_Dataset  


class TestVinDrCXRDataset(unittest.TestCase):
    def setUp(self):
        # create a temporary vin dr cxr structure
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        ann_dir = os.path.join(self.root, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root, "test"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "train"), exist_ok=True)

        # create minimal train/test CSVs with a single real class
        # here we pick "No finding" from your CLASSES list
        cols = ["image_id", "No finding"]
        df_train = pd.DataFrame([["img1", 1]], columns=cols).set_index("image_id")
        df_test  = pd.DataFrame([["img1", 1]], columns=cols).set_index("image_id")

        # Create all required annotation files
        df_train.to_csv(os.path.join(ann_dir, "image_labels_train.csv"))
        df_test.to_csv(os.path.join(ann_dir, "image_labels_test.csv"))

        # Create annotations files that filter_subset_annot_labels expects
        annot_cols = ["image_id", "class_name"]
        df_train_annot = pd.DataFrame([["img1", "No finding"]], columns=annot_cols)
        df_test_annot = pd.DataFrame([["img1", "No finding"]], columns=annot_cols)
        df_train_annot.to_csv(os.path.join(ann_dir, "annotations_train.csv"), index=False)
        df_test_annot.to_csv(os.path.join(ann_dir, "annotations_test.csv"), index=False)

        # also create the filtered files that your code will pick when class_labels is not None
        df_train.to_csv(os.path.join(ann_dir, "filtered_image_labels_train.csv"))
        df_test.to_csv(os.path.join(ann_dir, "filtered_image_labels_test.csv"))

        # dummy DICOM file (content doesn't matter since we'll mock dicom2array)
        open(os.path.join(self.root, "test", "img1.dicom"), "wb").close()
        open(os.path.join(self.root, "train", "img1.dicom"), "wb").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_split(self):
        # split must be "train" or "test"
        with self.assertRaises(AttributeError):
            VinDrCXR_Dataset(self.root, split="invalid", class_labels=None)

    def test_class_labels_none_assert(self):
        # your code asserts len(self.labels) == 15 when class_labels=None on train
        with self.assertRaises(AssertionError):
            VinDrCXR_Dataset(self.root, split="train", class_labels=None)

    @patch("rad_dino.data.VinDrCXR.data.filter_subset_annot_labels")
    @patch("rad_dino.data.VinDrCXR.data.dicom2array")
    def test_len_and_getitem(self, mock_dicom2array, mock_filter):
        # Setup mocks
        mock_filter.return_value = ["No finding"]
        mock_dicom2array.return_value = np.ones((5, 5), dtype=np.uint8)

        # normal usage on test split with a provided label list
        ds = VinDrCXR_Dataset(
            path_root=self.root,
            split="test",
            class_labels=["No finding"],
            transform=None
        )

        # only one image in our CSV
        self.assertEqual(len(ds), 1)

        img, target, image_id = ds[0]

        # dicom2array must have been called with the right path
        expected_path = os.path.join(self.root, "test", "img1.dicom")
        mock_dicom2array.assert_called_once_with(expected_path)

        # img comes back as the numpy array our mock returned
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(img.shape, (5, 5))

        # target is a torch.Tensor of shape [1] with value 1.0
        self.assertTrue(isinstance(target, torch.Tensor))
        np.testing.assert_array_equal(target.numpy(), np.array([1.0], dtype=np.float32))

        # image_id passes through
        self.assertEqual(image_id, "img1")

    @patch("rad_dino.data.VinDrCXR.data.filter_subset_annot_labels")
    @patch("rad_dino.data.VinDrCXR.data.dicom2array")
    def test_transform_applied(self, mock_dicom2array, mock_filter):
        # Setup mocks
        mock_filter.return_value = ["No finding"]
        mock_dicom2array.return_value = np.ones((3, 3), dtype=np.uint8)

        # test that the transform is applied
        transform = lambda x: x * 2  # double every pixel
        ds = VinDrCXR_Dataset(
            path_root=self.root,
            split="test",
            class_labels=["No finding"],
            transform=transform
        )

        img, _, _ = ds[0]
        # our mock returned ones, transform doubled → all 2s
        self.assertTrue((img == 2).all())


if __name__ == "__main__":
    unittest.main()
