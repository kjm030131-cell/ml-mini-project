import numpy as np
from scipy import ndimage
src=open('/home/claude/work/ML1/old.ipynb').read()
import json
nb=json.load(open('/home/claude/work/ML1/old.ipynb'))
c=''.join(nb['cells'][3]['source'])
# HOG 관련 함수 제외하고 실행
import re
parts=c.split('\ndef ')
keep=[p for p in parts if not p.startswith('hog')]
exec('\ndef '.join(keep))
def orient(X):  # 행 하나 단위: EMNIST 전치 + 0~1
    return np.ascontiguousarray((X/255.0).reshape(-1,28,28).transpose(0,2,1)).astype(np.float32)
