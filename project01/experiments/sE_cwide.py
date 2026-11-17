# 세션 E: C 넓은 범위 탐색 (gamma 1.2배 고정, 특성 pix1 h42 1 h84 2)
import sys; from common import run_svc
LOG = "../results/E_C_wide.md"; S = [("dkpix",1),("h42",1),("h84",2)]
Cs = [1, 3, 5, 20, 50, 100]
for i, C in enumerate(Cs):
    if i % 2 == int(sys.argv[1]): run_svc(f"C={C} gamma x1.2", S, LOG, C=C, gamma_mult=1.2)
