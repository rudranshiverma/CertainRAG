import json
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_rows(checkpoint_path):
    rows=[]
    with open(checkpoint_path) as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def calibrate(checkpoint_path):
    rows=load_rows(checkpoint_path)
    candidate_signals=["retrieval_confidence", "faithful_score", "self_consistency"]
    available=[s for s in candidate_signals if any(r.get(s) is not None for r in rows)]
    print(f"Signals found in checkpoint: {available}")
    if len(available)<2:
        print("Need at least 2 signals to calibrate.")
        return None

    X, y = [], []
    for r in rows:
        if any(r.get(s) is None for s in available):
            continue
        features=[]
        for s in available:
            val=float(r[s])
            if s == "self_consistency":
                val = 1.0-val
            features.append(val)
        X.append(features)
        y.append(1 if r["is_wrong"] else 0)

    X, y = np.array(X), np.array(y)
    print(f"Using {len(y)} rows — {y.sum()} wrong / {len(y) - y.sum()} correct")

    model=LogisticRegression()
    model.fit(X, y)

    raw_coefs=model.coef_[0]
    importance=np.abs(raw_coefs)
    total=importance.sum()
    if total==0:
        weights_raw={s: 1.0/len(available) for s in available}
    else:
        weights_raw = {s: float(importance[i]/total) for i, s in enumerate(available)}

    weight_key_map = {
        "retrieval_confidence": "retrieval_confidence",
        "faithful_score": "faithfulness",
        "self_consistency": "self_consistency",
    }
    final_weights={weight_key_map[k]: v for k, v in weights_raw.items()}

    print("\nCalibrated weights:")
    print(json.dumps(final_weights, indent=2))
    print("\nCopy into CertainRAG(weights={...}) or update DEFAULT_WEIGHTS in scorer.py")
    return final_weights


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/outputs_squad/checkpoint.jsonl"
    calibrate(path)