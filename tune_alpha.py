import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "competition_package"))
from utils import ScorerStepByStep

from solution import PredictionModel  # uses the ensemble above

test_file = os.path.join(CURRENT_DIR, "competition_package", "datasets", "valid_small.parquet")
scorer = ScorerStepByStep(test_file)

best = (-1e9, None)
for a in np.linspace(0.0, 1.0, 11):
    model = PredictionModel(alpha=float(a))
    res = scorer.score(model)
    print(a, res["weighted_pearson"])
    if res["weighted_pearson"] > best[0]:
        best = (res["weighted_pearson"], a)

print("BEST:", best)