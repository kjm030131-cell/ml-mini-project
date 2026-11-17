# 세션 C: 추가 특성 (강의 W06 31쪽 구멍 개수, 투영, 다른 크기 HOG)
import sys; from common import run_svc
LOG = "../results/C_extra_features.md"; B = [("dkpix",1),("h42",1),("h84",2)]
runs = [("기준 + 구멍 x1", B+[("holes",1)]),
        ("기준 + 구멍 x3", B+[("holes",3)]),
        ("기준 + 가로·세로 투영", B+[("rows",1),("cols",1)]),
        ("기준 + HOG 7칸/3이동", B+[("h73",1)]),
        ("기준 + HOG 6칸/2이동", B+[("h62",1)]),
        ("기준 + 원본 HOG(4,2) x0.5", B+[("rh42",0.5)])]
for i,(n,s) in enumerate(runs):
    if i % 2 == int(sys.argv[1]): run_svc(n, s, LOG)
