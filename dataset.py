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
        max_len_x = max(max_len_x, x.shape[-1])
        max_len_y = max(max_len_y, y.shape[-1])

    shape_x = [len(batch), *batch[0][0].shape]
    shape_y = [len(batch), *batch[0][1].shape]
    shape_x[-1] = max_len_x
    shape_y[-1] = max_len_y
    bx = np.zeros(shape_x, dtype=np.float32)
    by = np.zeros(shape_y, dtype=np.float32)
    lens = []
    for i, (x, y) in enumerate(batch):
        bx[i, ..., :x.shape[-1]] = x[...]
        by[i, ..., :y.shape[-1]] = y[...]
        lens.append([x.shape[-1], y.shape[-1]])

    return torch.as_tensor(bx), torch.as_tensor(by), lens

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
    def __init__(self, paths, u_size=16, t_size=64, n_fft=1024, hop_length=256):
        sps = []
        for p in paths:
            sps.extend(Path(p).glob('*'))
        self.pathss = [list(Path(p).glob('**/*.wav')) for p in sps]
        self.pathss = [ps for ps in self.pathss if len(ps)>=u_size]
        self.u_size = u_size
        self.segment_length = t_size*hop_length+n_fft-hop_length

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


class JvsParallelMgcc(torch.utils.data.Dataset):
    def __init__(self, root, sp_min=1, sp_max=90, ut_min=1, ut_max=90):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max

    def __len__(self):
        return 1 - self.ut_min + self.ut_max

    @property
    def n_speakers(self):
        return 1 - self.sp_min + self.sp_max

    def __getitem__(self, index):
        i = index + self.ut_min
        x_i, y_i = np.random.permutation(self.n_speakers)[:2]
        sp_x = 'jvs{:03}'.format(x_i + self.sp_min)
        sp_y = 'jvs{:03}'.format(y_i + self.sp_min)

        ut_path = 'VOICEACTRESS100_{:03}.mgcc.npy'.format(i)
        path_x = self.root / sp_x / 'parallel100/wav24kHz16bit' / ut_path
        path_y = self.root / sp_y / 'parallel100/wav24kHz16bit' / ut_path
        x = np.load(path_x)
        y = np.load(path_y)
        return x, y


class JvsNonparallelMgcc(torch.utils.data.Dataset):
    def __init__(self, root, u_size, t_size, sp_min=1, sp_max=90, ut_min=1, ut_max=90):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max
        self.u_size = u_size
        self.t_size = t_size

    @property
    def n_utterances(self):
        return 1 - self.ut_min + self.ut_max

    def __len__(self):
        return 1 - self.sp_min + self.sp_max

    def __getitem__(self, index):
        ids = np.random.permutation(self.n_utterances)[:self.u_size]
        
        mgccs = [self.load_mgcc(index, i) for i in ids]
        return np.asarray(mgccs)

    def load_mgcc(self, sp_i, ut_i):
        sp = 'jvs{:03}'.format(sp_i + self.sp_min)
        path = self.root / sp / 'parallel100/wav24kHz16bit' / 'VOICEACTRESS100_{:03}.mgcc.npy'.format(ut_i + self.ut_min)
        mgcc = np.load(path)
        if mgcc.shape[-1] >= self.t_size:
            max_start = mgcc.shape[-1] - self.t_size
            start = random.randint(0, max_start)
            mgcc = mgcc[:, start:start+self.t_size]
        else:
            offset = random.randint(0, self.t_size - mgcc.shape[-1])
            mgcc = np.pad(mgcc, [(0, 0), (offset, self.t_size - mgcc.shape[-1]-offset)], 'constant')
        return mgcc


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

    groups = torch.load('preprocess/text_group_list.pt')
    dataset = ParallelDataset('../data/vctk-preprocess/wavs/wav_16k/', groups)

    train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                              batch_size=4,
                              pin_memory=False,
                              drop_last=True, collate_fn=longest_pair_collate)

    for i, batch in enumerate(train_loader):
        x, y, lens = batch
        print(f'{i}: {x.shape}, {y.shape}')
        if i>10:
            break

    dataset = JvsParallelMgcc('../data/jvs/jvs_mgcc_16k_40/')

    train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                              batch_size=4,
                              pin_memory=False,
                              drop_last=True, collate_fn=longest_pair_collate)

    for i, batch in enumerate(train_loader):
        x, y, lens = batch
        print(f'{i}: {x.shape}, {y.shape}')
        if i>10:
            break

    dataset = JvsNonparallelMgcc('../data/jvs/jvs_mgcc_16k_40/', 8, 2048)
    train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                              batch_size=8,
                              pin_memory=False,
                              drop_last=True)

    for i, batch in enumerate(train_loader):
        print(f'{i}: {batch.shape}')
        if i>10:
            break