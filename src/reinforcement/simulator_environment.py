import torch
import torch.nn.functional as F
from torchrl.envs import EnvBase
from tensordict.tensordict import TensorDictBase, TensorDict
from typing import Optional, Type
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
    UnboundedDiscreteTensorSpec,
)
import json
from src.simulator.simulator import Simulator
from src.networks.encoders import GambleTripleEncoder


class SimulatorEnv(EnvBase):
    def __init__(self, simulator: Type[Simulator], encoder: GambleTripleEncoder, seed=None, device="cpu"):
        super().__init__(device=device, batch_size=[])

        self.load_settings()

        self.simulator = simulator()
        self.encoder = encoder
        self._make_specs()

        if seed is None:
            seed = torch.empty((), dtype=torch.long).random_().item()
        self.set_seed(seed)

    def load_settings(self):
        with open('configs/rl_settings.json') as f:
            settings = json.load(f)
        self.max_num_orders = settings['max_num_orders']
        self.max_num_couriers = settings['max_num_couriers']
        self.max_num_active_routes = settings['max_num_active_routes']

        with open('configs/network_hyperparams.json') as f:
            hyperparams = json.load(f)
        self.number_enc_dim = hyperparams['number_enc_dim']
        self.d_model = hyperparams['d_model']
        self.point_enc_dim = hyperparams['point_enc_dim']

    def make_masks(self, tensors):
        masks = {
            'o': torch.tensor([True] + [False] * (len(tensors['o']) - 1),
                              device=self.device, dtype=torch.bool),
            'c': torch.tensor([True] + [False] * (len(tensors['c']) - 1),
                              device=self.device, dtype=torch.bool),
            'ar': torch.tensor([True] + [False] * (len(tensors['ar']) - 1),
                               device=self.device, dtype=torch.bool)
        }

        return masks

    def pad_tensors(self, tensors, masks, ids):
        '''
        Pads tensors to max_limits inplace
        '''
        max_limits = {
            'o': self.max_num_orders,
            'c': self.max_num_couriers,
            'ar': self.max_num_active_routes
        }
        for item_type in ['o', 'c', 'ar']:
            length = tensors[item_type].shape[0]
            tensors[item_type] = F.pad(input=tensors[item_type],
                                       pad=(0, 0, 0, max_limits[item_type] - length),
                                       mode='constant', value=0.0)
            masks[item_type] = F.pad(input=masks[item_type],
                                     pad=(0, max_limits[item_type] - length),
                                     mode='constant', value=True)
            ids[item_type] = F.pad(input=ids[item_type],
                                   pad=(0, max_limits[item_type] - length),
                                   mode='constant', value=-1)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        '''
        tensordict['action'] - a np.array of indexes (not IDs) of couriers
        assigned for the given order.
        If there is no courier assigned self.max_num_couriers is provided.
        BOS-fake items are included.
        '''

        assignments = []
        assigned_o_idxs = set()
        assigned_c_idxs = set()
        for o_idx, c_idx in enumerate(tensordict['action'].detach().cpu().numpy()):
            if c_idx != self.max_num_couriers \
                and not tensordict['observation', 'masks', 'o'][o_idx] \
                and not tensordict['observation', 'masks', 'c'][c_idx] \
                and (o_idx not in assigned_o_idxs) \
                and (c_idx not in assigned_c_idxs):
                assignment = (
                    tensordict['observation', 'ids', 'o'][o_idx].item(),
                    tensordict['observation', 'ids', 'c'][c_idx].item()
                )
                assignments.append(assignment)
                assigned_o_idxs.add(o_idx)
                assigned_c_idxs.add(c_idx)

        # print(assignments)
        # print(self.simulator.GetState())
        prev_completed_orders = self.simulator.GetMetrics()['completed_orders']
        self.simulator.Next(assignments)
        triple = self.simulator.GetState()
        current_completed_orders = self.simulator.GetMetrics()['completed_orders']
        tensors, ids = self.encoder(triple, 0)
        masks = self.make_masks(tensors)
        self.pad_tensors(tensors, masks, ids)

        out = TensorDict(
            {
                "next": {
                    "observation": {
                        'tensors': {
                            'o': tensors['o'],
                            'c': tensors['c'],
                            'ar': tensors['ar']
                        },
                        'masks': {
                            'o': masks['o'],
                            'c': masks['c'],
                            'ar': masks['ar']
                        },
                        'ids': {
                            'o': ids['o'],
                            'c': ids['c'],
                            'ar': ids['ar']
                        }
                    },
                    "reward": torch.tensor(current_completed_orders - prev_completed_orders, dtype=torch.float32),
                    "done": torch.tensor(False, dtype=torch.bool),
                }
            },
            tensordict.shape
            # batch_size=tensordict.shape[0]
        )
        # print(out['next', 'observation', 'ids', 'o'])
        return out

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.simulator.reset()
        triple = self.simulator.GetState()
        tensors, ids = self.encoder(triple, 0)
        masks = self.make_masks(tensors)
        self.pad_tensors(tensors, masks, ids)
        new_observation = {
            'tensors': {
                'o': tensors['o'],
                'c': tensors['c'],
                'ar': tensors['ar']
            },
            'masks': {
                'o': masks['o'],
                'c': masks['c'],
                'ar': masks['ar']
            },
            'ids': {
                'o': ids['o'],
                'c': ids['c'],
                'ar': ids['ar']
            }
        }

        # if tensordict is None or tensordict.is_empty():
        return TensorDict(
            {
                "observation": new_observation
            },
            batch_size=self.batch_size
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_specs(self) -> None:
        self.action_spec = DiscreteTensorSpec(
            n=self.max_num_couriers + 1,
            dtype=torch.int,
            shape=[self.max_num_orders]
        )
        observation_spec = CompositeSpec(
            tensors=CompositeSpec(
                o=UnboundedContinuousTensorSpec(
                    shape=[self.max_num_orders, self.encoder.d_model],
                    dtype=torch.float
                ),
                c=UnboundedContinuousTensorSpec(
                    shape=[self.max_num_couriers, self.encoder.d_model],
                    dtype=torch.float
                ),
                ar=UnboundedContinuousTensorSpec(
                    shape=[self.max_num_active_routes, self.encoder.d_model],
                    dtype=torch.float
                ),
                # shape=[sub_batch_size]
            ),
            masks=CompositeSpec(
                o=DiscreteTensorSpec(
                    n=2,
                    dtype=torch.bool,
                    shape=[self.max_num_orders]
                ),
                c=DiscreteTensorSpec(
                    n=2,
                    dtype=torch.bool,
                    shape=[self.max_num_couriers]
                ),
                ar=DiscreteTensorSpec(
                    n=2,
                    dtype=torch.bool,
                    shape=[self.max_num_active_routes]
                ),
                # shape=[sub_batch_size]
            ),
            ids=CompositeSpec(
                o=UnboundedDiscreteTensorSpec(
                    dtype=torch.int,
                    shape=[self.max_num_orders]
                ),
                c=UnboundedDiscreteTensorSpec(
                    dtype=torch.int,
                    shape=[self.max_num_couriers]
                ),
                ar=UnboundedDiscreteTensorSpec(
                    dtype=torch.int,
                    shape=[self.max_num_active_routes]
                ),
                # shape=[sub_batch_size]
            ),
            # shape=[sub_batch_size]
        )
        # if not isinstance(observation_spec, CompositeSpec):
        observation_spec = CompositeSpec(observation=observation_spec)  # shape=[sub_batch_size]

        self.observation_spec = observation_spec
        self.reward_spec = UnboundedContinuousTensorSpec(
            # shape=[sub_batch_size],
            shape=[1],
            dtype=torch.float32,
        )
        self.done_spec = BinaryDiscreteTensorSpec(
            # n=sub_batch_size,
            # shape=[sub_batch_size],
            n=1,
            shape=[1],
            dtype=torch.bool
        )
