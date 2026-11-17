# R2: Stacking 재료 다양화 — R1(svc15+lgbm+raw+lr, KFold 2-fold OOF)에 성격이 다른 기본 모델 3개를 더해 본다
#   poly15 : 같은 svc15 특성, SVC 다항식 커널(3차)  → 커널이 달라 오답 패턴이 다름 (단독은 RBF보다 약했음)
#   svcH   : 실험 H 특성(기준 + 구멍 + HOG 7칸/3이동), RBF g1.2 → 다른 특성 관점의 SVC
#   et     : ExtraTrees(300그루) on svc15 특성 → 트리 계열, SVC와 오답이 가장 다를 것으로 예상
# 사용법 (experiments 폴더에서, R1 캐시를 공유):
#   python3 R1_stack_cv2.py prep && python3 R2_stack_diverse.py prep
#   python3 R1_stack_cv2.py svc15 | raw | lgbm | lr      (R1 기본 모델)
#   python3 R2_stack_diverse.py poly15 | svcH | et       (R2 새 기본 모델)
#   python3 R1_stack_cv2.py meta ; python3 R2_stack_diverse.py meta
# 규칙: 허용 모델(SVC, ExtraTrees, LGBM CPU, LogisticRegression, Stacking)만. KFold(비계층) OOF. 증강·PCA·StratifiedKFold 없음.
#       특성은 각 이미지 하나에서만 계산(레이블 미사용). 채점용 2만 개는 학습에 쓰지 않음.
import sys, os, time, numpy as np
from sklearn.model_selection import cross_val_predict
from common import block, labels
import R1_stack_cv2 as R1

OUT = R1.OUT; CV = R1.CV
LOG = "../results/R2_stack_diverse.md"
def log(s): print(s, flush=True); open(LOG, "a").write(s + "\n")

def X(name, part):
    if name in ("poly15", "et"): return R1.X("svc15", part)
    if name == "svcH": return block(part, [("dkpix", 1), ("h42", 1), ("h84", 2), ("holes", 1), ("h73", 1)])
    return R1.X(name, part)

def base(name):
    y, yte = labels(); A = X(name, "tr"); B = X(name, "te")
    from sklearn.svm import SVC
    if name == "poly15":
        m = SVC(C=10, kernel="poly", degree=3, coef0=1, gamma=1.0 / (A.shape[1] * A.var()), cache_size=500); method = "decision_function"
    elif name == "svcH":
        m = SVC(C=10, gamma=1.2 / (A.shape[1] * A.var()), cache_size=500); method = "decision_function"
    elif name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        A = A.astype(np.float32); B = B.astype(np.float32)
        m = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", n_jobs=2, random_state=42); method = "predict_proba"
    t = time.time(); o = cross_val_predict(m, A, y, cv=CV, method=method); t_oof = time.time() - t
    t = time.time(); m.fit(A, y); p = getattr(m, method)(B); t_fit = time.time() - t
    np.save(f"{OUT}/oof_{name}.npy", o); np.save(f"{OUT}/te_{name}.npy", p)
    w = int((p.argmax(1) != yte).sum())
    log(f"| {name} 단독 (OOF 2-fold {t_oof:.0f}s + 전체학습 {t_fit:.0f}s) | {w} | {1 - w / len(yte):.4f} | {t_oof + t_fit:.0f}s |")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep":
        if not os.path.exists(LOG): open(LOG, "w").write("| 실험 | 오답 (2만 중) | 정확도 | 시간 |\n|---|---|---|---|\n")
        for part in ("tr", "te"): X("svcH", part)
        print("prep done", flush=True)
    elif mode == "meta":
        from collections import Counter
        from sklearn.linear_model import LogisticRegression
        y, yte = labels()
        allv = [n for n in ["svc15", "lgbm", "raw", "lr", "poly15", "svcH", "et"] if os.path.exists(f"{OUT}/te_{n}.npy")]
        O = {n: np.load(f"{OUT}/oof_{n}.npy") for n in allv}; T = {n: np.load(f"{OUT}/te_{n}.npy") for n in allv}
        r1 = [n for n in ["svc15", "lgbm", "raw", "lr"] if n in allv]
        new = [n for n in ["poly15", "svcH", "et"] if n in allv]
        combos = [r1] + [r1 + [n] for n in new] + [allv] + [[n for n in allv if n != d] for d in r1 if d != "svc15"]
        seen = set()
        for c in combos:
            if tuple(c) in seen or len(c) < 2: continue
            seen.add(tuple(c))
            for C in (1, 10):
                t = time.time()
                m = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([O[k] for k in c]), y)
                p = m.predict(np.hstack([T[k] for k in c])); w = int((p != yte).sum())
                log(f"| Stacking {'+'.join(c)} (cv=2, LR C={C}) | {w} | {1 - w / len(yte):.4f} | {time.time() - t:.0f}s |")
        pair = Counter(tuple(sorted((int(a), int(b)))) for a, b in zip(yte[p != yte], p[p != yte])).most_common(6)
        log(f"\n마지막 조합(C=10) 오답 숫자쌍 상위: {pair}")
