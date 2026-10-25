"""
Lightweight librosa compatibility stub
"""

import numpy as np

__version__ = "0.10.1"

class feature:
    @staticmethod
    def melspectrogram(y=None, sr=22050, **kwargs):
        return np.zeros((43, 232), dtype=np.float32)

def load(path, sr=None, **kwargs):
    return np.zeros(44100, dtype=np.float32), 44100
