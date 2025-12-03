import yaml
from eda import run_eda
from utils import seed_everything

if __name__ == "__main__":
    # Load config
    with open("src/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get('random_seed', 42))
    segments, stats, matrix = run_eda(cfg)

    print("\n--- Phase 1 completed successfully ---")
    print("Saved outputs to:", cfg['paths']['outputs'])
