from pathlib import Path
import random

import librosa
import numpy as np
import numpy.random
import scipy.io.wavfile as wavfile
import torch
import torch.utils.data

class JvsAnalysis():
    def __init__(self, root, n_adapt=10, ut_min=91, ut_max=100):
        self.root = Path(root)
        self.adapt_wavs = [wavfile.read(p) for p in (self.root / 'nonpara30/wav24kHz16bit').glob('*.wav').sorted()[:n_adapt]]
        self.wavs = [wavfile.read(p) for p in [self.root / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{i:03}.wav' for i in range(ut_min, ut_max+1)]]