import unittest
from unittest.mock import patch

import numpy as np

from vimos.container import Photo


class TestPhoto(unittest.TestCase):
    def setUp(self):
        self.sample_path = "sample_path.jpg"
        self.sample_data = np.zeros((100, 100, 3))
        self.photo = Photo(self.sample_data)

    @patch("numpy.save")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_save(self, mock_exists, mock_makedirs, mock_save):
        save_path = "dir/sample.npy"
        self.photo.save(save_path)

        mock_exists.assert_called_once_with("dir")
        mock_makedirs.assert_called_once_with("dir")
        mock_save.assert_called_once_with(save_path, self.sample_data)

    def test_load(self):
        with patch("numpy.load", return_value=self.sample_data) as mock_load:
            self.photo.load(self.sample_path)
            mock_load.assert_called_once_with(self.sample_path)
            np.testing.assert_array_equal(self.photo.data, self.sample_data)

    # TODO: Add test for "apply"


if __name__ == "__main__":
    unittest.main()
