# 세션 A: 특성 묶음 비중 정밀 탐색  (사용법: python sA_weights.py 0 & python sA_weights.py 1)
import sys; from common import run_svc
LOG = "../results/A_feature_weights.md"
runs = [("pix1 h42 1 h84 2 (지난 최고, 재확인)", [("dkpix",1),("h42",1),("h84",2)]),
        ("pix1 h42 1 h84 3", [("dkpix",1),("h42",1),("h84",3)]),
        ("pix1 h42 1 h84 4", [("dkpix",1),("h42",1),("h84",4)]),
        ("pix0.7 h42 1 h84 2", [("dkpix",0.7),("h42",1),("h84",2)]),
        ("pix1 h42 0.8 h84 2", [("dkpix",1),("h42",0.8),("h84",2)]),
        ("pix1 h42 1.2 h84 2", [("dkpix",1),("h42",1.2),("h84",2)])]
for i,(n,s) in enumerate(runs):
    if i % 2 == int(sys.argv[1]): run_svc(n, s, LOG)
