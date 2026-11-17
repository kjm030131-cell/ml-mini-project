# 세션 G: HOG 계산 방식 개선 (부호 있는 방향 18구간, 가우시안 흐림, Dalal-Triggs 블록 정규화)
# 모두 C=10, gamma 1.2배. 기준 = 실험 B 최고 (오답 95)
import sys; from common import run_svc
LOG = "../results/G_hog_variants.md"
runs = [("기준 HOG(4,2)+HOG(8,4)x2", [("dkpix",1),("h42",1),("h84",2)]),
        ("부호 있는 방향 18구간", [("dkpix",1),("h42s",1),("h84s",2)]),
        ("가우시안 흐림 0.7 후 HOG", [("dkpix",1),("h42b",1),("h84b",2)]),
        ("흐림 + 부호 있는 방향", [("dkpix",1),("h42sb",1),("h84sb",2)]),
        ("Dalal-Triggs 블록 HOG(4칸)+(7칸)x2", [("dkpix",1),("hdt4",1),("hdt7",2)]),
        ("기준 + DT 블록 HOG(4칸, 흐림)", [("dkpix",1),("h42",1),("h84",2),("hdt4b",1)])]
for i,(n,s) in enumerate(runs):
    if i % 2 == int(sys.argv[1]): run_svc(n, s, LOG, C=10, gamma_mult=1.2)
