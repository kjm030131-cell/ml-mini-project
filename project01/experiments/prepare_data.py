# emnist-digits-train.csv -> emnist.npy (실험 스크립트들이 읽는 형식)
import numpy as np, pandas as pd
d = pd.read_csv("emnist-digits-train.csv", header=None, dtype=np.uint8).to_numpy()
print(d.shape, np.bincount(d[:, 0]))
np.save("emnist.npy", d)
