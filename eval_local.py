"""
Local evaluation script.

Usage:
    python eval_local.py                      # scores solution.py (default)
    python eval_local.py solution_adaptive    # scores solution_adaptive.py
    python eval_local.py solution_gru         # scores solution_gru.py
    python eval_local.py --small              # use valid_small.parquet (faster)

The script imports PredictionModel from the given module, runs it through
ScorerStepByStep on the local valid.parquet, and prints scores.
"""
import sys
import os
import time
import importlib.util

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VALID_FULL  = os.path.join(REPO_ROOT, "competition_package", "datasets", "valid.parquet")
VALID_SMALL = os.path.join(REPO_ROOT, "competition_package", "datasets", "valid_small.parquet")
UTILS_PATH  = os.path.join(REPO_ROOT, "competition_package", "utils.py")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "competition_package"))


def load_solution(module_name: str):
    """Load PredictionModel from a .py file in the repo root."""
    # Strip .py extension if given
    module_name = module_name.removesuffix(".py")
    py_path = os.path.join(REPO_ROOT, module_name + ".py")
    if not os.path.exists(py_path):
        print(f"ERROR: {py_path} not found")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    # Make sure the module can find checkpoints in REPO_ROOT
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.PredictionModel


def main():
    args = sys.argv[1:]
    use_small = "--small" in args
    args = [a for a in args if a != "--small"]

    module_name = args[0] if args else "solution"
    valid_path = VALID_SMALL if use_small else VALID_FULL

    print(f"Solution : {module_name}.py")
    print(f"Data     : {os.path.basename(valid_path)}")
    print()

    # Load scorer
    scorer_spec = importlib.util.spec_from_file_location("utils", UTILS_PATH)
    scorer_mod = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(scorer_mod)
    ScorerStepByStep = scorer_mod.ScorerStepByStep

    # Load model class
    PredictionModel = load_solution(module_name)

    print("Loading model...")
    model = PredictionModel()

    print("Scoring...")
    t0 = time.time()
    scorer = ScorerStepByStep(valid_path)
    results = scorer.score(model)
    elapsed = time.time() - t0

    print()
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k:<25} {v:.6f}")
    print("=" * 40)
    print(f"  Time: {elapsed:.1f}s")
    print()

    wp = results.get("weighted_pearson", 0)
    print(f">>> weighted_pearson = {wp:.6f}")


if __name__ == "__main__":
    main()
