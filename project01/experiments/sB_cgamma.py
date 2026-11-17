# 세션 B: C·gamma 정밀 탐색 (특성 비중 pix1 h42 1 h84 2 기준)
import sys; from common import run_svc
LOG = "../results/B_C_gamma.md"; S = [("dkpix",1),("h42",1),("h84",2)]
runs = [(7,1.0),(15,1.0),(10,0.8),(10,1.2),(15,1.2),(7,1.4)]
for i,(C,g) in enumerate(runs):
    if i % 2 == int(sys.argv[1]): run_svc(f"C={C} gamma x{g}", S, LOG, C=C, gamma_mult=g)
