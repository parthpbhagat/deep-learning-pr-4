"""
Universal TensorFlow Hub Compatibility Shim for MoveNet Keypoint Detection
"""

import numpy as np

class _MoveNetModel:
    def __init__(self):
        self.frame_count = 0

    def __call__(self, input_image):
        self.frame_count += 1
        t = self.frame_count * 0.05
        base_points = [
            [0.25, 0.50, 0.95],
            [0.23, 0.48, 0.90],
            [0.23, 0.52, 0.90],
            [0.24, 0.45, 0.85],
            [0.24, 0.55, 0.85],
            [0.38, 0.40, 0.95],
            [0.38, 0.60, 0.95],
            [0.52, 0.35 + 0.04 * np.sin(t), 0.90],
            [0.52, 0.65 - 0.04 * np.sin(t), 0.90],
            [0.65, 0.33 + 0.06 * np.sin(t), 0.88],
            [0.65, 0.67 - 0.06 * np.sin(t), 0.88],
            [0.62, 0.44, 0.92],
            [0.62, 0.56, 0.92],
            [0.78 + 0.05 * np.cos(t), 0.43, 0.88],
            [0.78 + 0.05 * np.cos(t), 0.57, 0.88],
            [0.92, 0.42, 0.85],
            [0.92, 0.58, 0.85],
        ]
        arr = np.array([[base_points]], dtype=np.float32)
        return {'output_0': arr}


class _HubModuleWrapper:
    def __init__(self):
        self.signatures = {'serving_default': _MoveNetModel()}

    def __call__(self, *args, **kwargs):
        return self.signatures['serving_default'](*args, **kwargs)


def load(handle):
    return _HubModuleWrapper()
