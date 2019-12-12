from pathlib import Path
import random

import librosa
import numpy as np
import numpy.random
import scipy.io.wavfile as wavfile
import torch
import torch.utils.data

def infinit_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def longest_pair_collate(batch):
    max_len_x, max_len_y = 0, 0
    for x, y in batch:
        max_len_x = max(max_len_x, len(x))
        max_len_y = max(max_len_y, len(y))

    bx = np.zeros([len(batch), max_len_x], dtype=np.float32)
    by = np.zeros([len(batch), max_len_y], dtype=np.float32)
    for i, (x, y) in enumerate(batch):
        bx[i, :len(x)] = x[...]
        by[i, :len(y)] = y[...]

    return bx, by

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, roots, segment_length=16384):
        self.paths = []
        for f in roots:
            self.paths.extend(Path(f).glob('**/*.wav'))
        self.segment_length = segment_length
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sr, audio_np = wavfile.read(self.paths[index])
        if audio_np.shape[0] >= self.segment_length:
            max_audio_start = audio_np.shape[0] - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio_np = audio_np[audio_start:audio_start+self.segment_length]
        else:
            max_pad = self.segment_length - audio_np.shape[0]
            pad_start = random.randint(0, max_pad)
            audio_np = np.pad(audio_np, (pad_start, max_pad - pad_start), 'constant')
        return (audio_np/32768.0).astype(np.float32)


class MultiSpeakerAudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, u_size=16, segment_length=16384+1024-256):
        sps = []
        for p in paths:
            sps.extend(Path(p).glob('*'))
        self.pathss = [list(Path(p).glob('**/*.wav')) for p in sps]
        self.pathss = [ps for ps in self.pathss if len(ps)>=u_size]
        self.u_size = u_size
        self.segment_length = segment_length

    def __len__(self):
        return len(self.pathss)

    def __getitem__(self, idx):
        paths = self.pathss[idx]
        ii = np.random.permutation(len(paths))[:self.u_size]
        return np.asarray([self.load_wav(paths[i]) for i in ii])

    def load_wav(self, path):
        sr, audio_np = wavfile.read(path)
        assert(len(audio_np.shape)==1)
        if audio_np.shape[0] >= self.segment_length:
            max_audio_start = audio_np.shape[0] - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio_np = audio_np[audio_start:audio_start+self.segment_length]
        else:
            offset = random.randint(0, self.segment_length - audio_np.shape[0])
            audio_np = np.pad(audio_np, (offset, self.segment_length - audio_np.shape[0]-offset), 'constant')
        return (audio_np/32768.0).astype(np.float32)


class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, root, groups):
        self.root = root
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = [self.root/p.with_suffix('.wav') for p in self.groups[index]]
        idx = np.random.permutation(len(group))[:2]
        px, py = group[idx[0]], group[idx[1]]
        x = wavfile.read(px)[1]/32768.0
        y = wavfile.read(py)[1]/32768.0
        return x.astype(np.float32), y.astype(np.float32)

if __name__ == "__main__":
    # test
    from torch.utils.data import DataLoader
    dataset = AudioDataset(['../data/voiceactors/voice_actors_split_wav_16k'])
    train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                              batch_size=10,
                              pin_memory=False,
                              drop_last=True)
    for i, batch in enumerate(train_loader):
        audio = batch
        print(f'{i}: {audio.shape}')
        if i>10:
            break

    if not Path('text_group_list.pt').exists():
        import preprocess
        preprocess.extract_vctk_para('../data/VCTK/VCTK-Corpus/txt/')

    groups = torch.load('text_group_list.pt')
    dataset = ParallelDataset('../data/vctk-preprocess/wavs/wav_16k/', groups)

    train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                              batch_size=4,
                              pin_memory=False,
                              drop_last=True, collate_fn=longest_pair_collate)

    for i, batch in enumerate(train_loader):
        x, y = batch
        print(f'{i}: {x.shape}, {y.shape}')
        if i>10:
            break