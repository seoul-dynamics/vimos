import unittest
from unittest.mock import patch

import numpy as np

from vimos.container import Photo, Album


class TestAlbum(unittest.TestCase):
    def setUp(self):
        self.sample_data = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]
        self.album = Album(self.sample_data)

    @patch("numpy.save")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_save(self, mock_exists, mock_makedirs, mock_save):
        sample_path = "dir/sample.npy"
        with patch.object(Album, "_prepare_data", return_value=self.sample_data):
            self.album.save(sample_path)

        mock_exists.assert_called_once_with("dir")
        mock_makedirs.assert_called_once_with("dir")
        mock_save.assert_called_once()

    @patch("numpy.load", return_value=np.zeros((2, 100, 100, 3)))
    def test_load(self, mock_load):
        sample_path = "sample.npy"
        self.album.load(sample_path)
        mock_load.assert_called_once_with(sample_path)
        self.assertEqual(len(self.album), 2)
        self.assertIsInstance(self.album[0], Photo)

    # TODO: Add test for "apply"
