# R1: 16차 최고 Stacking(svc15+lgbm+raw+lr, 오답 78, 내부 cv=3)을 내부 cv=2로 줄여도 성능이 유지되는지 검증
# 목적: 24만 개 10-fold 전체 실행 시간을 줄이기 위한 설정 찾기 (RESEARCH.md 아이디어 대기열 0번)
# 사용법: python3 R1_stack_cv2.py prep            (특성 캐시를 먼저 만든다. 동시 실행 시 캐시 충돌 방지)
#         python3 R1_stack_cv2.py svc15 | raw | lgbm | lr   (2-fold OOF + 전체 학습 → 홀드아웃 점수 저장)
#         python3 R1_stack_cv2.py meta            (저장된 결과로 LogisticRegression 최종 판정)
# 규칙: 허용 모델(SVC, LGBM CPU, LogisticRegression, Stacking)만 사용. KFold(비계층) OOF, 증강·PCA·StratifiedKFold 없음.
import sys, os, time, itertools, numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from common import feat, block, labels, _imgs, load_raw, CACHE, TR, TE
from feats import holes, hog2, build3

OUT = "../results/R1_cache"; os.makedirs(OUT, exist_ok=True)
LOG = "../results/R1_stack_cv2.md"
CV = KFold(n_splits=2, shuffle=False)  # cross_val_predict 기본(cv=정수)은 분류에서 StratifiedKFold라서 직접 KFold 지정
def log(s): print(s, flush=True); open(LOG, "a").write(s + "\n")

def _cached(part, key, fn):
    p = os.path.join(CACHE, f"{part}_{key}.npy")
    if not os.path.exists(p): np.save(p, fn().astype(np.float32))
    return np.load(p)
def extra14(part):  # 14차: 행·열 획 전환 횟수 + 위쪽/왼쪽 잉크 비율 + 구멍 (보정 이미지 기준)
    def f():
        im = feat(part, "dkpix").reshape(-1, 28, 28); b = (im > 0.3).astype(np.int8)
        tr_r = (np.diff(b, axis=2) == 1).sum(2) / 4.0; tr_c = (np.diff(b, axis=1) == 1).sum(1) / 4.0
        s = im.sum((1, 2)).clip(1e-6)
        top = im[:, :10, :].sum((1, 2)) / s; left = im[:, :, :14].sum((1, 2)) / s
        h = holes(im) / np.array([1, 50], np.float32)
        return np.hstack([tr_r, tr_c, top[:, None], left[:, None], h])
    return _cached(part, "R1_extra14", f)
def profiles(part):  # 15차: 행·열별 첫/마지막 획 위치(윤곽선) 112개
    def f():
        b = feat(part, "dkpix").reshape(-1, 28, 28) > 0.3
        first = lambda m, ax: np.where(m.any(ax), np.argmax(m, ax), 28) / 28.0
        return np.hstack([first(b, 2), 1 - first(b[:, :, ::-1], 2), first(b, 1), 1 - first(b[:, ::-1, :], 1)])
    return _cached(part, "R1_prof", f)
def rh84(part): return _cached(part, "R1_rh84", lambda: hog2(_imgs(part, False), 8, 4))
def lgbm_x(part):
    def f():
        d = load_raw(); sl = TR if part == "tr" else TE; raw = np.asarray(d[sl, 1:])
        return np.vstack([build3(raw[s:s + 10000]) for s in range(0, len(raw), 10000)])
    return _cached(part, "R1_build3", f)

def X(name, part):
    if name == "svc15": return np.hstack([block(part, [("dkpix", 1), ("h42", 1), ("h84", 2)]), extra14(part) * 1.5, profiles(part)]).astype(np.float64)
    if name == "raw": return np.hstack([block(part, [("rawpix", 1), ("rh42", 1)]), rh84(part)]).astype(np.float64)
    if name == "lr": return block(part, [("dkpix", 1), ("h42", 1), ("h84", 1)])
    if name == "lgbm": return lgbm_x(part)

def base(name):
    y, yte = labels(); A = X(name, "tr"); B = X(name, "te")
    if name in ("svc15", "raw"):
        from sklearn.svm import SVC
        gm = 1.2 if name == "svc15" else 1.0
        m = SVC(C=10, gamma=gm / (A.shape[1] * A.var()), cache_size=500); method = "decision_function"
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=300, learning_rate=0.1, max_bin=63, colsample_bytree=0.3, n_jobs=1, random_state=42, verbose=-1); method = "predict_proba"
    else:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=10, max_iter=300); method = "predict_proba"
    t = time.time(); o = cross_val_predict(m, A, y, cv=CV, method=method); t_oof = time.time() - t
    t = time.time(); m.fit(A, y); p = getattr(m, method)(B); t_fit = time.time() - t
    np.save(f"{OUT}/oof_{name}.npy", o); np.save(f"{OUT}/te_{name}.npy", p)
    w = int((p.argmax(1) != yte).sum())
    log(f"| {name} 단독 (OOF 2-fold {t_oof:.0f}s + 전체학습 {t_fit:.0f}s) | {w} | {1 - w / len(yte):.4f} | {t_oof + t_fit:.0f}s |")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep":
        if not os.path.exists(LOG): open(LOG, "w").write("| 실험 | 오답 (2만 중) | 정확도 | 시간 |\n|---|---|---|---|\n")
        for part in ("tr", "te"):
            for k in ("dkpix", "h42", "h84", "rawpix", "rh42"): feat(part, k)
            extra14(part); profiles(part); rh84(part); lgbm_x(part)
        print("prep done", flush=True)
    elif mode == "meta":
        from sklearn.linear_model import LogisticRegression
        y, yte = labels()
        names = [n for n in ["svc15", "lgbm", "raw", "lr"] if os.path.exists(f"{OUT}/te_{n}.npy")]
        O = {n: np.load(f"{OUT}/oof_{n}.npy") for n in names}; T = {n: np.load(f"{OUT}/te_{n}.npy") for n in names}
        for r in range(2, len(names) + 1):
            for c in itertools.combinations(names, r):
                if "svc15" not in c: continue
                for C in (1, 10):
                    t = time.time()
                    m = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([O[k] for k in c]), y)
                    w = int((m.predict(np.hstack([T[k] for k in c])) != yte).sum())
                    log(f"| Stacking {'+'.join(c)} (cv=2, LR C={C}) | {w} | {1 - w / len(yte):.4f} | {time.time() - t:.0f}s |")
    else:
        base(mode)
