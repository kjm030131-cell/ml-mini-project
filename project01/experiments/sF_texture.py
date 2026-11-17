# 세션 F: 질감 특성 LBP·가보르 추가 (기준: pix1 h42 1 h84 2, C=10, gamma 1.2배 = 실험 B 최고 95개)
import sys; from common import run_svc
LOG = "../results/F_texture.md"; B = [("dkpix",1),("h42",1),("h84",2)]
runs = [("기준 (B 최고 재확인)", B, 1.2),
        ("기준 + LBP", B+[("lbp",1)], 1.2),
        ("기준 + 가보르", B+[("gab",1)], 1.2),
        ("기준 + LBP + 가보르", B+[("lbp",1),("gab",1)], 1.2),
        ("기준 + LBP x2 + 가보르 x2", B+[("lbp",2),("gab",2)], 1.2),
        ("기준 + LBP + 가보르, gamma 1.0", B+[("lbp",1),("gab",1)], 1.0)]
for i,(n,s,g) in enumerate(runs):
    if i % 2 == int(sys.argv[1]): run_svc(n, s, LOG, C=10, gamma_mult=g)
