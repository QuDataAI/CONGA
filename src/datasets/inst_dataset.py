from pathlib import Path
import numpy as np


class InstDataset:
    def __init__(self, folder) -> None:
        self.path = f'{folder}'
        self.files  = sorted([f for f in Path(self.path).iterdir() if f.is_file()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        name = fname.name.split('/')[-1]
        with open(fname, "r") as f:
            lst = [ [float(v) for v in st.split(' ') ] for st in f if len(st)]

            n, w_max = lst[0][0], lst[0][1]
            if len(lst[-1]) > 2:
                x = lst[-1]
                lst = np.array(lst[1:-1]).transpose()
            else:
                x = None
                lst = np.array(lst[1:]).transpose()
            v, w = list(lst[0]), list(lst[1])

            assert (x is None or n==len(x)) and n==len(v) and n==len(w), f"Wrong data: n={n}, len(x)={None if x is None else len(x)}, len(v)={len(v)}, len(w)={len(w)}"
            if x:
                v_max = np.array(x) @ np.array(v)
                w_tot = np.array(x) @ np.array(w)
                assert w_tot <= w_max,  f"Wrong data: w_tot={w_tot}, w_max={w_max}"
            else:
                v_max = None
                # open optimum file
                fname_split = str(fname).split('/')
                fname_opt = "/".join(fname_split[:-1]) + "-optimum/" + fname_split[-1]
                with open(fname_opt, "r") as f_opt:
                    v_max = float(f_opt.read())

            return   v, w, w_max, v_max, name
