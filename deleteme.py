import json
import torch
import random
import numpy

from src.reinforcement.base import Runner, InferenceMetricsRunner, MetricLogger
from src.reinforcement.delivery import DeliveryMaker


torch.manual_seed(seed=0)
numpy.random.seed(seed=0)
random.seed(0)
with open('configs/training.json') as f:
    cfg = json.load(f)
    cfg['use_wandb'] = False
    cfg['device'] = 'cpu'
    cfg['fix_zero_seed'] = True
with open('configs/paths.json') as f:
    cfg2 = json.load(f)
    cfg.update(cfg2)
maker = DeliveryMaker(**cfg)
maker.actor_critic.load_state_dict(torch.load('checkpoints/4e9b39ab6eb443b5b2febfb0b73f4597.pt', map_location='cpu'))
runner = maker.sampler.runner
metric_logger = maker.metric_logger
inference = InferenceMetricsRunner(runner, metric_logger)

if __name__ == '__main__':
    inference()
