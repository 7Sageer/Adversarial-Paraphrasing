import unittest

import torch

from runtime_utils import resolve_runtime_device, resolve_torch_dtype


class RuntimeUtilsTest(unittest.TestCase):
    def test_auto_device_falls_back_to_cpu_when_accelerators_are_unavailable(self):
        original_cuda = torch.cuda.is_available
        original_mps = torch.backends.mps.is_available
        try:
            torch.cuda.is_available = lambda: False
            torch.backends.mps.is_available = lambda: False
            self.assertEqual(resolve_runtime_device("auto"), "cpu")
        finally:
            torch.cuda.is_available = original_cuda
            torch.backends.mps.is_available = original_mps

    def test_float16_on_cpu_fails_fast(self):
        with self.assertRaises(RuntimeError):
            resolve_torch_dtype("cpu", "float16")


if __name__ == "__main__":
    unittest.main()
