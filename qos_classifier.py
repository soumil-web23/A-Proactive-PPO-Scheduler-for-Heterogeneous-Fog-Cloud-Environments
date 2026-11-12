"""
qos_classifier.py
==================
The "Ingestion Gateway" from Section 4.1 of the report: a Decision Tree
that labels every incoming task Delay-Tolerant (0) or Delay-Sensitive (1)
using deadline, payload size and sensor origin.

Implementation note (an honest deviation from the report you should
mention in your paper): scikit-learn's DecisionTreeClassifier does not
implement textbook ID3 (which only supports categorical multi-way splits
and never prunes). We use CART with the `entropy` criterion, which is
the standard, actively-maintained, information-gain-driven equivalent
used in nearly all applied ID3-labelled work, and it supports the mixed
categorical/continuous features this task needs (deadline, payload size)
without manual discretisation. If a reviewer specifically wants textbook
ID3, `id3_from_scratch()` below is a small pure-Python fallback that
implements true information-gain splitting on discretised bins so you
can report both and show they agree.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data_gen import SENSOR_TYPES, synthetic_task_stream

CRITICAL_SENSORS = {"av_lidar", "health_monitor"}


def label_priority(df: pd.DataFrame) -> pd.Series:
    """Ground-truth QoS priority used only to TRAIN the classifier
    (in a real deployment this would come from historical SLA logs).
    Rule: tight deadline OR safety-critical sensor -> Delay-Sensitive.
    """
    tight_deadline = df["deadline_ms"] < 80
    critical_sensor = df["sensor_type"].isin(CRITICAL_SENSORS)
    return (tight_deadline | critical_sensor).astype(int)


class QoSClassifier:
    """Wraps a one-hot encoder + entropy-criterion decision tree."""

    def __init__(self, max_depth: int = 5, random_state: int = 42):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.tree = DecisionTreeClassifier(
            criterion="entropy", max_depth=max_depth, random_state=random_state
        )
        self.numeric_cols = ["deadline_ms", "payload_kb"]
        self.cat_col = "sensor_type"
        self._fitted = False

    def _features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cat = df[[self.cat_col]]
        if fit:
            cat_enc = self.encoder.fit_transform(cat)
        else:
            cat_enc = self.encoder.transform(cat)
        num = df[self.numeric_cols].to_numpy()
        return np.hstack([num, cat_enc])

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X = self._features(df, fit=True)
        self.tree.fit(X, y)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("QoSClassifier.fit() must be called first")
        X = self._features(df, fit=False)
        return self.tree.predict(X)

    def predict_one(self, sensor_type: str, deadline_ms: float, payload_kb: float) -> int:
        row = pd.DataFrame([{
            "sensor_type": sensor_type, "deadline_ms": deadline_ms, "payload_kb": payload_kb
        }])
        return int(self.predict(row)[0])


def train_qos_classifier(n_samples: int = 4000, test_size: float = 0.2, seed: int = 42):
    df = synthetic_task_stream(n_samples, seed=seed)
    y = label_priority(df)
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = QoSClassifier().fit(train_df, y_train)
    preds = clf.predict(test_df)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"[QoSClassifier] held-out accuracy={acc:.4f}  f1={f1:.4f}")
    return clf, acc, f1


# ---- optional textbook-ID3 fallback, for reviewers who want "real" ID3 ----
def id3_from_scratch(df: pd.DataFrame, y: pd.Series, n_bins: int = 4):
    """A minimal, pure-Python ID3 (categorical multiway splits, no pruning)
    over discretised features. Returns a nested-dict tree and a predict fn.
    Included for completeness / reviewer transparency, not used by the
    main pipeline because it cannot handle unseen continuous values well.
    """
    work = df.copy()
    for col in ["deadline_ms", "payload_kb"]:
        work[col] = pd.qcut(work[col], q=n_bins, duplicates="drop").astype(str)
    work["label"] = y.values
    feature_cols = ["sensor_type", "deadline_ms", "payload_kb"]

    def entropy(labels):
        p = labels.value_counts(normalize=True)
        return -(p * np.log2(p + 1e-12)).sum()

    def build(data, features):
        labels = data["label"]
        if labels.nunique() == 1:
            return {"leaf": int(labels.iloc[0])}
        if not features:
            return {"leaf": int(labels.mode()[0])}
        base_h = entropy(labels)
        gains = {}
        for f in features:
            weighted_h = 0.0
            for val, sub in data.groupby(f):
                weighted_h += (len(sub) / len(data)) * entropy(sub["label"])
            gains[f] = base_h - weighted_h
        best = max(gains, key=gains.get)
        node = {"split": best, "children": {}, "majority": int(labels.mode()[0])}
        for val, sub in data.groupby(best):
            node["children"][val] = build(sub, [f for f in features if f != best])
        return node

    tree = build(work, feature_cols)

    def predict_row(row, node):
        if "leaf" in node:
            return node["leaf"]
        val = row[node["split"]]
        child = node["children"].get(val)
        if child is None:
            return node["majority"]
        return predict_row(row, child)

    return tree, predict_row, work


if __name__ == "__main__":
    train_qos_classifier()
