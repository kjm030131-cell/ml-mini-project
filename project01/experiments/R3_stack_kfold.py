# R3: 제출 노트북 Stacking 을 StratifiedKFold 없이(KFold 객체) 돌려도 오답 80 근처가 유지되는지 검증
# 배경: 제출 노트북(mini_proj_01_20221329_stack.ipynb)은 StackingClassifier(cv=2) 와 cross_val_score(cv=10) 처럼
#       정수 cv 를 쓴다. 사이킷런은 분류기에 정수 cv 를 주면 자동으로 StratifiedKFold 를 쓰므로 규칙 위반 위험.
#       → 내부 cv 를 KFold(n_splits=2) 로 바꿔도 성능이 같은지 5만/2만 기준으로 확인한다.
# 사용: python3 R3_stack_kfold.py kfold2      (KFold 2-fold, 섞지 않음)
#       python3 R3_stack_kfold.py kfold2s     (KFold 2-fold, shuffle=True, random_state=42)
#       python3 R3_stack_kfold.py smoke       (1500/600 개로 빠른 동작 확인, 결과 파일에 안 씀)
import json, os, sys, time, resource
import numpy as np
from sklearn.model_selection import KFold
from common import load_raw, TR, TE, ROOT

mode = sys.argv[1]
NB = os.path.join(ROOT, "..", "mini_proj_01_20221329_stack.ipynb")
OUT = os.path.join(ROOT, "..", "results", "R3_stack_kfold.md")

nb = json.load(open(NB))
src = ["".join(c["source"]) for c in nb["cells"]]
d = load_raw()
if mode == "smoke":
    tr, te = slice(0, 1500), slice(220000, 220600)
else:
    tr, te = TR, TE
X_tr = np.asarray(d[tr, 1:]).astype(np.float32); y_tr = np.asarray(d[tr, 0]).copy()
X_te = np.asarray(d[te, 1:]).astype(np.float32); y_te = np.asarray(d[te, 0]).copy()

# 노트북 셀 3(특성 함수), 4(by_chunk), 5(모델 정의) 를 그대로 실행. 셀 5 는 X_train[:5] 로 모양을 출력하므로 X_train 필요
g = {"np": np, "X_train": X_tr, "y_train": y_tr}
exec("import numpy as np\nimport pandas as pd", g)
for i in (3, 4, 5):
    exec(src[i], g)
stack = g["stack_clf"]
if mode in ("kfold2", "smoke"):
    stack.set_params(cv=KFold(n_splits=2))
elif mode == "kfold2s":
    stack.set_params(cv=KFold(n_splits=2, shuffle=True, random_state=42))
else:
    raise SystemExit("unknown mode " + mode)
print("cv =", stack.cv, flush=True)

t = time.time()
stack.fit(X_tr, y_tr)
p = stack.predict(X_te)
sec = time.time() - t
wrong = int((p != y_te).sum())
mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
label = {"kfold2": "KFold(2)", "kfold2s": "KFold(2, shuffle, rs=42)", "smoke": "KFold(2) smoke"}[mode]
line = f"| 노트북 Stacking, 내부 cv={label} | {wrong} | {1 - wrong / len(y_te):.4f} | {sec / 60:.0f}분 | {mem}MB |"
print("RESULT", line, flush=True)

# 오답 숫자쌍 상위 (정답→예측)
pairs = {}
for a, b in zip(y_te[p != y_te], p[p != y_te]):
    pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1
top = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
print("PAIRS", top, flush=True)

if mode != "smoke":
    new = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if new:
            f.write("| 실험 | 오답 (2만 중) | 정확도 | 시간 | 최대 메모리 |\n|---|---|---|---|---|\n")
        f.write(line + "\n")
        f.write(f"<!-- {mode} 오답 숫자쌍 상위(정답,예측):개수 {top} -->\n")
