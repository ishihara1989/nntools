import sys
import pathlib

import numpy as np
import pysptk
import pyworld
import scipy.io.wavfile as wavfile


def mcep_dir(srcroot, tgtroot, n_mcep=40, alpha=0.42):
    src = pathlib.Path(srcroot)
    tgt = pathlib.Path(tgtroot)
    if not pathlib.Path(src).exists():
        raise ValueError('src not exists: {}'.format(src))

    for p in sorted(src.glob('**/*.wav')):
        print(p)
        tgt_dir = tgt / p.parent.relative_to(src)
        tgt_stem = (tgt_dir / p.name).with_suffix('')
        tgt_dir.mkdir(parents=True, exist_ok=True)
        mcep_path = tgt_stem.with_suffix('.mcep.npy')
        c0_path = tgt_stem.with_suffix('.c0.npy')
        f0_path = tgt_stem.with_suffix('.f0.npy')
        if mcep_path.exists() and c0_path.exists() and f0_path.exists():
            print('skip')
            continue

        sr, wav = wavfile.read(p)
        x = (wav/32768.0).astype(np.float64)
        f0, sp, ap = pyworld.wav2world(x.astype(np.float64), sr)
        mfcc = pysptk.sp2mc(sp, n_mcep, alpha)
        f0, mfcc = f0.astype(np.float32), mfcc.T.astype(np.float32)
        c0 = mfcc[0, :]
        mfcc = np.ascontiguousarray(mfcc[1:, :])

        np.save(mcep_path, mfcc)
        np.save(c0_path, c0)
        np.save(f0_path, f0)
        print(tgt_stem, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} srcdir tgtdir [alpha]')
        exit(1)
    if len(sys.argv) >= 4:
        try:
            alpha = float(sys.argv[3])
        except:
            print(f'failed to parse as float: {sys.argv[3]}')
            exit(1)
    else:
        alpha = 0.42
    mcep_dir(sys.argv[1], sys.argv[2], alpha=alpha)