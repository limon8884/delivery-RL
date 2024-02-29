import json

from src_new.reinforcement.delivery import run_ppo


if __name__ == '__main__':
    with open('configs_new/training.json') as f:
        cfg = json.load(f)
    run_ppo(**cfg)
