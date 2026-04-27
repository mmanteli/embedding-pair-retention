from data_and_encoder import read_precalculated_embeddings
from linearity_metrics import evaluate_pairset
import json
from dataclasses import asdict
import os
import sys
import numpy as np

if len(sys.argv) == 1: #
    print("Running a random test")
    X = np.random.random((800, 1024))
    Y = np.random.random((800, 1024))
    r = evaluate_pairset(X,Y)
    print(asdict(r))
    exit()

embedding_column_names = ("emb1", "emb2")
embs = sys.argv[1]
assert os.path.exists(embs), "No embedding file found."
X, Y, ds, _ = read_precalculated_embeddings(embs, embedding_column_names)
#if len(sys.argv) < 2:
save_path = embs.replace("embeddings", "knn_results", 1).replace(".pkl", ".json")
#else:
#    save_path = sys.argv[2]
os.makedirs(os.path.dirname(save_path), exist_ok=True)
r = evaluate_pairset(X,Y)

def save_report_json(report, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)

print(asdict(r))
save_report_json(r, save_path)