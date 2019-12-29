import sys
import pathlib

import numpy as np
import pysptk
import pyworld
import scipy.io.wavfile as wavfile


def resample_dir(srcroot, tgtroot, n_mgcc=40, gamma=0.42):
    src = pathlib.Path(srcroot)
    tgt = pathlib.Path(tgtroot)
    if not pathlib.Path(src).exists():
        raise ValueError('src not exists: {}'.format(src))

    for p in sorted(src.glob('**/*.wav')):
        print(p)
        sr, wav = wavfile.read(p)
        x = (wav/32768.0).astype(np.float64)
        f0, sp, ap = pyworld.wav2world(x.astype(np.float64), sr)
        mfcc = pysptk.sp2mc(sp, n_mgcc, gamma)
        f0, mfcc = f0.astype(np.float32), mfcc.T.astype(np.float32)
        c0 = mfcc[0, :]
        mfcc = np.ascontiguousarray(mfcc[1:, :])

        tgt_dir = tgt / p.parent.relative_to(src)
        tgt_stem = (tgt_dir / p.name).with_suffix('')
        tgt_dir.mkdir(parents=True, exist_ok=True)

        np.save(tgt_stem.with_suffix('.mgcc.npy'), mfcc)
        np.save(tgt_stem.with_suffix('.c0.npy'), c0)
        np.save(tgt_stem.with_suffix('.f0.npy'), f0)
        print(tgt_stem, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} srcdir tgtdir')
        exit(1)
    resample_dir(sys.argv[1], sys.argv[2])