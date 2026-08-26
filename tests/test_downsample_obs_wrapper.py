import math
import unittest
from unittest.mock import patch

import torch

from utils import DownsampleObsWrapper


class DownsampleObsWrapperTest(unittest.TestCase):
    def test_downsamples_when_only_width_matches_target(self):
        for shape in ((8, 4, 3), (2, 8, 4, 3)):
            with self.subTest(shape=shape):
                wrapper = object.__new__(DownsampleObsWrapper)
                wrapper.target_size = 4
                rgb = torch.arange(math.prod(shape)).reshape(shape).to(torch.uint8)

                obs = wrapper.observation({"rgb": rgb})

                self.assertEqual(obs["rgb"].shape, (*shape[:-3], 4, 4, 3))
                self.assertEqual(obs["rgb"].dtype, torch.uint8)

    def test_skips_interpolation_when_both_dimensions_match(self):
        wrapper = object.__new__(DownsampleObsWrapper)
        wrapper.target_size = 4
        rgb = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)

        with patch("utils.F.interpolate") as interpolate:
            obs = wrapper.observation({"rgb": rgb})

        interpolate.assert_not_called()
        self.assertIs(obs["rgb"], rgb)


if __name__ == "__main__":
    unittest.main()
