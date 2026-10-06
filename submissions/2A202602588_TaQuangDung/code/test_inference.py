"""Focused checks for inference transforms and numerics."""
import unittest

import numpy as np
import torch
from torch import nn

import inference


class InferenceTests(unittest.TestCase):
    def test_views_and_aggregation(self):
        x = torch.arange(16).reshape(1, 1, 4, 4)
        self.assertTrue(torch.equal(inference.view_hflip(x), x.flip(-1)))
        self.assertEqual(len(inference.views_multicrop(x, 2)), 5)
        self.assertEqual([v.shape[-1] for v in inference.views_multiscale(x.float(), [2, 8])], [2, 8])
        logits = np.array([[3.0, 0.0], [0.0, 3.0]])
        np.testing.assert_allclose(inference.aggregate_views([logits, logits]),
                                   inference.apply_temperature(logits, 1.0), atol=1e-12)

    def test_temperature_preserves_prediction(self):
        logits = np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 2.0]])
        labels = np.array([0, 1, 1])
        temperature = inference.fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        np.testing.assert_array_equal(inference.apply_temperature(logits, temperature).argmax(1),
                                      logits.argmax(1))

    def test_conv_bn_fusion_preserves_output(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU()).eval()
        images = torch.randn(2, 3, 8, 8)
        with torch.inference_mode():
            expected = model(images)
            fused = inference.fuse_conv_bn(model)
            actual = fused(images)
        self.assertEqual(fused.fused_conv_bn_count, 1)
        self.assertLess((actual - expected).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
