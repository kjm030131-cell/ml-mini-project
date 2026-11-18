# R6: Stacking 최종 판정기(메타 모델) 비교 + 골격 특성 SVC(R4 x1.5, 단독 85)를 재료로 교체
# 배경: 단일 SVC 특성 추가는 85~92 에서 포화(R4), 데이터 양도 병목 아님(R5). 재료 추가도 78~81 정체(17차).
#       남은 축은 '최종 판정기'. 지금까지 LR(C=1/10) 만 써 봄. W15 허용 목록의 Stacking 은 final_estimator 를
#       허용 모델 중 아무거나 쓸 수 있으므로 LR C 넓게, LGBM(CPU), SVC(RBF), ExtraTrees 를 메타로 비교한다.
# 방식: R1 과 같은 수동 Stacking. 기본 모델 OOF 는 KFold(2) 객체(비계층, 섞지 않음) → StratifiedKFold 아님.
#       메타 모델 튜닝은 OOF(학습 5만) 안에서만 하고, 채점 2만은 보고용으로만 쓴다(채점 데이터로 학습 안 함).
# 사용: python3 R6_meta.py prep | svc15 | lgbm | raw | lr | svcR4 | meta | smoke
# 규칙: SVC, LGBM CPU, LogisticRegression, ExtraTrees, Stacking 만. 증강·PCA·StratifiedKFold 없음. 레이블은 특성 계산에 안 씀.
import sys, os, time, itertools, numpy as np
import R1_stack_cv2 as r1
from common import labels, ROOT

r1.OUT = os.path.join(ROOT, "..", "results", "R6_cache"); os.makedirs(r1.OUT, exist_ok=True)
r1.LOG = os.path.join(ROOT, "..", "results", "R6_meta.md")
OUT, LOG = r1.OUT, r1.LOG

_X = r1.X
def X(name, part):
    if name == "svcR4":  # R4 sk15: svc15 특성 + 골격 끝점·갈림점 20개 x1.5 (R4 단독 85)
        from R4_skeleton import skel_feats
        return np.hstack([_X("svc15", part), skel_feats(part) * 1.5]).astype(np.float64)
    return _X(name, part)
r1.X = X

_base = r1.base
def base(name):
    if name == "svcR4":  # 골격 특성 SVC: OOF(decision_function) + 전체 학습
        if True:
            from sklearn.svm import SVC
            y, yte = labels(); A = X("svcR4", "tr"); B = X("svcR4", "te")
            m = SVC(C=10, gamma=1.2 / (A.shape[1] * A.var()), cache_size=500)
            from sklearn.model_selection import cross_val_predict
            t = time.time(); o = cross_val_predict(m, A, y, cv=r1.CV, method="decision_function"); t1 = time.time() - t
            t = time.time(); m.fit(A, y); p = m.decision_function(B); t2 = time.time() - t
            np.save(f"{OUT}/oof_svcR4.npy", o); np.save(f"{OUT}/te_svcR4.npy", p)
            w = int((p.argmax(1) != yte).sum())
            r1.log(f"| svcR4 단독 (OOF 2-fold {t1:.0f}s + 전체학습 {t2:.0f}s) | {w} | {1 - w / len(yte):.4f} | {t1 + t2:.0f}s |")
        return
    _base(name)

def metas():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    return {
        "LR C=0.1": lambda: LogisticRegression(C=0.1, max_iter=3000),
        "LR C=1": lambda: LogisticRegression(C=1, max_iter=3000),
        "LR C=10": lambda: LogisticRegression(C=10, max_iter=3000),
        "LR C=100": lambda: LogisticRegression(C=100, max_iter=3000),
        "SVC rbf C=1": lambda: SVC(C=1, gamma="scale"),
        "SVC rbf C=10": lambda: SVC(C=10, gamma="scale"),
        "LGBM 200": lambda: LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40, n_jobs=2, random_state=42, verbose=-1),
        "ExtraTrees 300": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=2, random_state=42),
    }

def meta(smoke=False):
    from sklearn.model_selection import KFold, cross_val_predict
    from collections import Counter
    y, yte = labels()
    names = [n for n in ["svc15", "svcR4", "lgbm", "raw", "lr"] if os.path.exists(f"{OUT}/te_{n}.npy")]
    O = {n: np.load(f"{OUT}/oof_{n}.npy") for n in names}; T = {n: np.load(f"{OUT}/te_{n}.npy") for n in names}
    combos = [c for c in (("svc15", "lgbm", "raw", "lr"), ("svcR4", "lgbm", "raw", "lr"),
                          ("svc15", "svcR4", "lgbm", "raw", "lr")) if all(k in names for k in c)]
    inner = KFold(n_splits=5, shuffle=True, random_state=0)  # 메타 선택용: 학습 5만 OOF 안에서만 평가
    for c in combos:
        A = np.hstack([O[k] for k in c]); B = np.hstack([T[k] for k in c])
        if smoke: A, yy, B, yt = A[:3000], y[:3000], B[:1000], yte[:1000]
        else: yy, yt = y, yte
        for mn, mk in metas().items():
            t = time.time()
            ins = int((cross_val_predict(mk(), A, yy, cv=inner) != yy).sum())  # 학습 5만 안 메타 오답
            m = mk().fit(A, yy); pr = m.predict(B); w = int((pr != yt).sum())
            pairs = Counter((int(a), int(b)) for a, b in zip(yt[pr != yt], pr[pr != yt])).most_common(3)
            line = f"| Stacking {'+'.join(c)} / 최종 {mn} | {w} | {1 - w / len(yt):.4f} | {time.time() - t:.0f}s | 학습내 OOF 오답 {ins}/{len(yy)} {pairs} |"
            print(line, flush=True)
            if not smoke: open(LOG, "a").write(line + "\n")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep":
        if not os.path.exists(LOG):
            open(LOG, "w").write("| 실험 | 오답 (2만 중) | 정확도 | 시간 | 비고 |\n|---|---|---|---|---|\n")
        for part in ("tr", "te"):
            for k in ("dkpix", "h42", "h84", "rawpix", "rh42"): r1.feat(part, k)
            r1.extra14(part); r1.profiles(part); r1.rh84(part); r1.lgbm_x(part)
            from R4_skeleton import skel_feats; skel_feats(part)
        print("prep done", flush=True)
    elif mode == "meta": meta()
    elif mode == "smoke":  # 가짜 OOF 로 메타 단계만 동작 확인
        y, yte = labels(); rng = np.random.default_rng(0)
        for n in ["svc15", "lgbm", "raw", "lr"]:
            np.save(f"{OUT}/oof_{n}.npy", np.eye(10)[y] + rng.normal(0, .5, (len(y), 10)))
            np.save(f"{OUT}/te_{n}.npy", np.eye(10)[yte] + rng.normal(0, .5, (len(yte), 10)))
        meta(smoke=True)
        for n in ["svc15", "lgbm", "raw", "lr"]:
            os.remove(f"{OUT}/oof_{n}.npy"); os.remove(f"{OUT}/te_{n}.npy")
    else: base(mode)
