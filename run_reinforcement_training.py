import json

from src.reinforcement.delivery import run_ppo


if __name__ == '__main__':
    with open('configs/training.json') as f:
        cfg = json.load(f)
    run_ppo(**cfg)
