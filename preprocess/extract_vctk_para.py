import pathlib

import torch

def extract_vctk_para(root):
    root = pathlib.Path(root)
    txts = {}
    for sp in root.glob('*'):
        for fp in sp.glob('*.txt'):
            print(fp.relative_to(root))
            with fp.open() as f:
                txt = f.read().strip()
            l = txts.get(txt, [])
            l.append(fp.relative_to(root).with_suffix(''))
            txts[txt] = l
    torch.save(txts, 'text_dict.pt')
    path_list = [v for v in txts.values() if len(v)>1]
    torch.save(path_list, 'text_group_list.pt')


if __name__ == "__main__":
    extract_vctk_para('../../data/VCTK/VCTK-Corpus/txt/')