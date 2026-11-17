# 세션 H: 효과 있던 것 합치기 — 구멍 개수(C·메인 92), HOG 7칸/3이동(C 92), gamma 1.2배(B 95)
import sys; from common import run_svc
LOG = "../results/H_combine.md"; B = [("dkpix",1),("h42",1),("h84",2)]
runs = [("기준 + 구멍 + HOG73, g1.2", B+[("holes",1),("h73",1)], 10, 1.2),
        ("기준 + 구멍 x1.5 + HOG73, g1.2", B+[("holes",1.5),("h73",1)], 10, 1.2),
        ("기준 + 구멍 + HOG73, g1.2, C=5", B+[("holes",1),("h73",1)], 5, 1.2),
        ("기준 + 구멍 + HOG73 x2, g1.2", B+[("holes",1),("h73",2)], 10, 1.2),
        ("기준 + 구멍 + HOG73, g1.0", B+[("holes",1),("h73",1)], 10, 1.0),
        ("기준 + 구멍 + HOG73 + 원본HOG x0.5, g1.2", B+[("holes",1),("h73",1),("rh42",0.5)], 10, 1.2)]
for i,(n,s,C,g) in enumerate(runs):
    if i % 2 == int(sys.argv[1]): run_svc(n, s, LOG, C=C, gamma_mult=g)
