import json

from src.reinforcement.base import Runner, InferenceMetricsRunner, MetricLogger
from src.reinforcement.delivery import DeliveryMaker


with open('configs/training.json') as f:
    cfg = json.load(f)
    cfg['use_wandb'] = False
    cfg['device'] = 'cpu'
    cfg['fix_zero_seed'] = True
with open('configs/paths.json') as f:
    cfg2 = json.load(f)
    cfg.update(cfg2)
maker = DeliveryMaker(**cfg)
runner = maker.sampler.runner
metric_logger = maker.metric_logger
inference = InferenceMetricsRunner(runner, metric_logger)
inference()
print(metric_logger.logs)