"""Rebuild the small artifacts the Streamlit app needs (run once, locally, after downloading the data).

    python app/build_artifacts.py

Writes: app/model/xgb_higgs.json, app/data/oof_scores.npz, app/data/sample_events.csv, app/data/meta.json
and copies the result tables. Same settings as the notebook: 30 raw features, -999 -> NaN,
class-balanced weights, XGBoost (300 trees, lr 0.08, depth 6), seed 42, stratified 5-fold CV.
"""
import json, os, shutil, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
import higgs_utils as hu

SEED, FRAC_FALLBACK = 42, 0.15
params = dict(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", n_jobs=-1, random_state=SEED)
ids, X, y, w = hu.load_train(f"{ROOT}/data/training.csv")
Xn = hu.to_nan(X); N = len(y)

oof = np.zeros(N)
for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xn, y):
    m = xgb.XGBClassifier(**params).fit(Xn.iloc[a], y[a], sample_weight=hu.class_balanced_weights(y[a], w[a]))
    oof[b] = m.predict_proba(Xn.iloc[b])[:, 1]
bt = hu.best_threshold(y, oof, w)
frac = bt["frac"]
k = int(frac * N) - 1
thr = float(np.sort(oof)[::-1][k])
print("pooled OOF AMS", round(bt["ams"], 3), "at fraction", frac, "| probability cut", round(thr, 4))

final = xgb.XGBClassifier(**params).fit(Xn, y, sample_weight=hu.class_balanced_weights(y, w))
final.save_model(f"{ROOT}/app/model/xgb_higgs.json")
np.savez_compressed(f"{ROOT}/app/data/oof_scores.npz", score=oof.astype("float32"), y=y.astype("int8"), w=w.astype("float32"))

_, Xt = hu.load_test(f"{ROOT}/data/test.csv")
Xt.sample(300, random_state=SEED).to_csv(f"{ROOT}/app/data/sample_events.csv", index=False)
json.dump({"fraction": frac, "prob_threshold": thr, "oof_ams": bt["ams"], "n_oof": N, "features": list(X.columns)},
          open(f"{ROOT}/app/data/meta.json", "w"), indent=1)
for f in ("experiments_validation.csv", "cv_summary.csv", "cv_folds.csv", "results.json"):
    shutil.copy(f"{ROOT}/results/{f}", f"{ROOT}/app/data/{f}")

# consistency check with the submission produced by the notebook (rank-based top-15 % rule)
sub_path = f"{ROOT}/submission/submission.csv"
if os.path.exists(sub_path):
    s = pd.read_csv(sub_path); sc = final.predict_proba(hu.to_nan(Xt))[:, 1]
    top = np.argsort(np.argsort(sc)) + 1 > int(round(len(sc) * (1 - frac)))
    print("agreement of re-saved model with notebook submission:", float((top == (s.Class.to_numpy() == "s")).mean()))
