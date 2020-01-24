from pathlib import Path
import random

import numpy as np
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

def longest_quad_collate(batch):
    batch = [tp for tp in batch if tp is not None]
    if len(batch) == 0: return None
    maxs = [0, 0, 0, 0]
    for tp in batch:
        for i in range(4):
            maxs[i] = max(maxs[i], tp[i].shape[-1])
    shapes = [[len(batch), *batch[0][i].shape] for i in range(4)]
    for i in range(4):
        shapes[i][-1] = maxs[i]
    ret = [np.zeros(s, dtype=np.float32) for s in shapes]
    for b, tp in enumerate(batch):
        for i in range(4):
            ret[i][b, ..., :tp[i].shape[-1]] = tp[i][...]
    return [torch.as_tensor(r) for r in ret]


def load_mgcc(path, t_size):
        mgcc = np.load(path)
        if mgcc.shape[-1] >= t_size:
            max_start = mgcc.shape[-1] - t_size
            start = random.randint(0, max_start)
            mgcc = mgcc[:, start:start+t_size]
        else:
            offset = random.randint(0, t_size - mgcc.shape[-1])
            mgcc = np.pad(mgcc, [(0, 0), (offset, t_size - mgcc.shape[-1]-offset)], 'constant')
        return mgcc


class RandomPicker():
    def __init__(self, data, n=1):
        self.data = data
        self.index = 0
        self.shuffle()
        self.n = n

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        self.permutation = np.random.permutation(len(self))

    def pick(self):
        if self.index > len(self)-self.n:
            self.shuffle()
            self.index = 0
        idx = self.permutation[self.index:self.index+self.n]
        self.index += 1
        if self.n == 1:
            return self.data[idx[0]]
        else:
            return [self.data[i] for i in idx]


class RandomPair():
    def __init__(self, data, n=1):
        self.data = data
        self.index = 0
        self.shuffle()
        self.n = n

    def __len__(self):
        n = len(self.data)
        return n * (n-1) // 2

    def unfold_index(self, k, N):
        i = k // N
        j = k % N
        if i >= j:
            return i+1, j
        else:
            return N-i-1, N-j-1

    def shuffle(self):
        self.permutation = np.random.permutation(len(self))

    def pick(self):
        if self.index >= len(self):
            self.shuffle()
            self.index = 0
        idx = self.permutation[self.index]
        self.index += 1
        i, j = self.unfold_index(idx, len(self.data))
        return self.data[i], self.data[j]


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

    def map_path(self, index, ut_path):
        i = index + self.sp_min
        sp = 'jvs{:03}'.format(i)
        return self.root / sp / 'parallel100/wav24kHz16bit' / ut_path

    def __getitem__(self, index):
        i = index + self.ut_min
        ut_path = 'VOICEACTRESS100_{:03}.mcep.npy'.format(i)

        path_x, path_y = [self.map_path(idx, ut_path) for idx in np.random.permutation(self.n_speakers) if self.map_path(idx, ut_path).exists()][:2]
        x = np.load(path_x)
        y = np.load(path_y)
        return x, y


class JvsNonparallelMcep(torch.utils.data.Dataset):
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
        sp = 'jvs{:03}'.format(index + self.sp_min)
        paths = [self.map_path(i, sp) for i in np.random.permutation(self.n_utterances) if self.map_path(i, sp).exists()][:self.u_size]
        
        mgccs = [self.load_mgcc(p) for p in paths]
        return np.asarray(mgccs)

    def map_path(self, index, sp):
        i = index + self.ut_min
        return self.root / sp / 'parallel100/wav24kHz16bit' / 'VOICEACTRESS100_{:03}.mcep.npy'.format(i)

    def load_mgcc(self, path):
        mgcc = np.load(path)
        if mgcc.shape[-1] >= self.t_size:
            max_start = mgcc.shape[-1] - self.t_size
            start = random.randint(0, max_start)
            mgcc = mgcc[:, start:start+self.t_size]
        else:
            offset = random.randint(0, self.t_size - mgcc.shape[-1])
            mgcc = np.pad(mgcc, [(0, 0), (offset, self.t_size - mgcc.shape[-1]-offset)], 'constant')
        return mgcc

class JvsTwoParallelMcep(torch.utils.data.Dataset):
    def __init__(self, root, sp_min=1, sp_max=90, ut_min=1, ut_max=90, verbose=False):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max
        self.verbose = verbose

    @property
    def n_speakers(self):
        return 1 - self.sp_min + self.sp_max

    @property
    def n_utterances(self):
        return 1 - self.ut_min + self.ut_max

    def __len__(self):
        return self.n_speakers * (self.n_speakers-1) * self.n_utterances * (self.n_utterances-1) // 4

    def indeces(self, index):
        J = self.n_utterances * (self.n_utterances-1) // 2
        sk = index // J
        uk = index % J
        si, sj = self.unfold_index(sk, self.n_speakers)
        ui, uj = self.unfold_index(uk, self.n_utterances)
        return si, sj, ui, uj

    def unfold_index(self, k, N):
        # index to combination
        i = k // N
        j = k % N
        if i >= j:
            return i+1, j
        else:
            return N-i-1, N-j-1

    def map_path(self, sp, ut):
        i = sp + self.sp_min
        j = ut + self.ut_min
        return self.root / f'jvs{i:03}' / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{j:03}.mcep.npy'

    def __getitem__(self, index):
        si, sj, ui, uj = self.indeces(index)
        if self.verbose:
            print(self.sp_min+si, self.sp_min+sj, self.ut_min+ui, self.ut_min+uj)
        x1 = self.map_path(si, ui)
        x2 = self.map_path(si, uj)
        y1 = self.map_path(sj, ui)
        y2 = self.map_path(sj, uj)
        paths = [x1, x2, y1, y2]
        if not all([p.exists() for p in paths]):
            return None

        return [np.load(p) for p in paths]

class SourceFilterDataset(torch.utils.data.Dataset):
    def __init__(self, audio_root, feature_root, sp_min=1, sp_max=90, ut_min=1, ut_max=90):
        self.audio_root = Path(audio_root)
        self.feature_root = Path(feature_root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max
        sps = [p for p in self.audio_root.glob('jvs*') if p.name <= f'jvs{sp_max:03}' and p.name >= f'jvs{sp_min:03}']
        audios = []
        for sp in sps:
            audios.extend([p for p in sp.glob('parallel100/wav24kHz16bit/VOICEACTRESS100_*.wav') if p.name <= f'VOICEACTRESS100_{ut_max:03}.wav' and p.name >= f'VOICEACTRESS100_{ut_min:03}.wav'])
        self.audios = audios

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        audio_path = self.audios[index]
        rel = audio_path.relative_to(self.audio_root)
        audio = wavfile.read(audio_path)[1]/32768.0
        mcep = np.load(self.feature_root / rel.with_suffix('.mcep.npy'))
        c0 = np.load(self.feature_root / rel.with_suffix('.c0.npy'))
        f0 = np.load(self.feature_root / rel.with_suffix('.f0.npy'))
        ap = np.load(self.feature_root / rel.with_suffix('.ap.npy'))
        return audio, f0, c0, mcep, ap

class VCTKParallel(torch.utils.data.Dataset):
    def __init__(self, root, sp_min=225, sp_max=343, ut_min=9, ut_max=24, verbose=False):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max
        self.verbose = verbose
        self.parallels = []
        for i in range(ut_min, ut_max+1):
            self.parallels.append(sorted([p for p in self.root.glob(f'*/*_{i:03}.mcep.npy') if p.name >= f'p{sp_min:03}_{i:03}.mcep.npy' and p.name <= f'p{sp_max:03}_{i:03}.mcep.npy']))
        self.parallel_pickers = [RandomPair(p) for p in self.parallels]
        self.nonparallels = {p.name: RandomPicker([pp for pp in p.glob('*.mcep.npy') if pp.name > f'{p.name}_{ut_max:03}.mcep.npy']) for p in self.root.glob(f'*')}

    def __len__(self):
        return len(self.parallels)

    def __getitem__(self, index):
        px, py = self.parallel_pickers[index].pick()
        npx = self.nonparallels[px.parent.name].pick()
        npy = self.nonparallels[py.parent.name].pick()
        return np.load(px), np.load(npx), np.load(py), np.load(npy)


class VCTKNonParallel(torch.utils.data.Dataset):
    def __init__(self, root, sp_min=225, sp_max=343, ut_min=9, u_size=8, t_size=512, verbose=False):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.u_size = u_size
        self.t_size = t_size
        self.verbose = verbose
        self.nonparallels = [RandomPicker([pp for pp in p.glob('*.mcep.npy') if pp.name >= f'{p.name}_{ut_min:03}.mcep.npy'], n=u_size) for p in self.root.glob(f'*')]

    def __len__(self):
        return len(self.nonparallels)

    def __getitem__(self, index):
        paths = self.nonparallels[index].pick()
        return np.asarray([load_mgcc(p, self.t_size) for p in paths])


if __name__ == "__main__":
    # test
    from torch.utils.data import DataLoader
    import sys
    if len(sys.argv)>1:
        kind = sys.argv[1]
    else:
        kind = 'all'

    is_all = kind == 'all'

    if kind == 'audio' or is_all:
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

    if kind == 'parallel' or is_all:
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

    if kind == 'jvsparallel' or is_all:
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

    if kind == 'jvsnonpara' or is_all:
        dataset = JvsNonparallelMcep('../data/jvs/jvs_mgcc_16k_40/', 8, 2048)
        train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                                batch_size=8,
                                pin_memory=False,
                                drop_last=True)

        for i, batch in enumerate(train_loader):
            print(f'{i}: {batch.shape}')
            if i>10:
                break

    if kind == 'jvstwo' or is_all:
        dataset = JvsTwoParallelMcep('../data/jvs/jvs_mgcc_16k_40/', sp_min=1, sp_max=4, ut_min=1, ut_max=5, verbose=True)
        train_loader = DataLoader(dataset, num_workers=1, shuffle=False,
                                batch_size=1,
                                pin_memory=False,
                                drop_last=False, collate_fn=longest_quad_collate)
        for i, batch in enumerate(train_loader):
            if i>100:
                break

    if kind == 'vctkpara' or is_all:
        dataset = VCTKParallel('../data/vctk-preprocess/mcep/')
        train_loader = DataLoader(dataset, num_workers=1, shuffle=False,
                                batch_size=4,
                                pin_memory=False,
                                drop_last=False, collate_fn=longest_quad_collate)
        for i, batch in enumerate(train_loader):
            px, npx, py, npy = batch
            print(px.size(), py.size(), npx.size(), npy.size())
            if i>10:
                break

    if kind == 'vctknonpara' or is_all:
        dataset = VCTKNonParallel('../data/vctk-preprocess/mcep/')
        train_loader = DataLoader(dataset, num_workers=8, shuffle=False,
                                batch_size=8,
                                pin_memory=False,
                                drop_last=False)

        for i, batch in enumerate(train_loader):
            print(batch.size())
            if i>10:
                break