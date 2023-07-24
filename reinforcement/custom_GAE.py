import torch
import torch.nn as nn
from torchrl.objectives.utils import hold_out_net
from torchrl.envs.utils import step_mdp
from typing import Union, Tuple, List, Optional
from torchrl.modules import SafeModule
from torchrl.objectives.value.utils import _custom_conv1d, _make_gammas_tensor
from tensordict.tensordict import TensorDictBase


class CustomGAE(nn.Module):
    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: float,
        value_network: SafeModule,
        average_gae: bool = False,
        differentiable: bool = False,
        advantage_key: Union[str, Tuple] = "advantage",
        value_target_key: Union[str, Tuple] = "value_target",
        value_key: Union[str, Tuple] = "state_value",
    ):
        super().__init__()
        try:
            device = next(value_network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=device))
        self.value_network = value_network
        self.value_key = value_key
        if value_key not in value_network.out_keys:
            raise KeyError(
                f"value key '{value_key}' not found in value network out_keys."
            )

        self.average_gae = average_gae
        self.differentiable = differentiable

        self.advantage_key = advantage_key
        self.value_target_key = value_target_key

        self.in_keys = (
            value_network.in_keys
            + [("next", "reward"), ("next", "done")]
            + [("next", in_key) for in_key in value_network.in_keys]
        )
        self.out_keys = [self.advantage_key, self.value_target_key]

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[torch.Tensor]] = None,
        target_params: Optional[List[torch.Tensor]] = None,
    ) -> TensorDictBase:
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", "reward"))
        gamma, lmbda = self.gamma, self.lmbda
        kwargs = {}
        if self.is_functional and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if params is not None:
            kwargs["params"] = params
        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            self.value_network(tensordict, **kwargs)

        value = tensordict.get(self.value_key)

        step_td = step_mdp(tensordict)
        if target_params is not None:
            # we assume that target parameters are not differentiable
            kwargs["params"] = target_params
        elif "params" in kwargs:
            kwargs["params"] = kwargs["params"].detach()
        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            self.value_network(step_td, **kwargs)
        next_value = step_td.get(self.value_key)
        done = tensordict.get(("next", "done"))
               
        adv, value_target = self.vec_generalized_advantage_estimate(
            gamma, lmbda, value, next_value, reward, done
        )

        if self.average_gae:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            adv = adv - loc
            adv = adv / scale

        tensordict.set(self.advantage_key, adv)
        tensordict.set(self.value_target_key, value_target)

        return tensordict
    
    def vec_generalized_advantage_estimate(self, 
        gamma: float,
        lmbda: float,
        state_value: torch.Tensor, # [*bs, ts, 100]
        next_state_value: torch.Tensor, # [*bs, ts, 100]
        reward: torch.Tensor, # [*bs, ts, 1]
        done: torch.Tensor, # [*bs, ts, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get generalized advantage estimate of a trajectory.

        Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
        https://arxiv.org/pdf/1506.02438.pdf for more context.

        Args:
            gamma (scalar): exponential mean discount.
            lmbda (scalar): trajectory discount.
            state_value (Tensor): value function result with old_state input.
                must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
            next_state_value (Tensor): value function result with new_state input.
                must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
            reward (Tensor): reward of taking actions in the environment.
                must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
            done (Tensor): boolean flag for end of episode.

        """
        # for tensor in (next_state_value, state_value, reward, done):
        #     if tensor.shape[-1] != 1:
        #         raise RuntimeError(
        #             "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
        #         )
        state_value = state_value.unsqueeze(-3).transpose(-1, -3)
        next_state_value = next_state_value.unsqueeze(-3).transpose(-1, -3)
        reward = reward.unsqueeze(-3).transpose(-1, -3)
        done = done.unsqueeze(-3).transpose(-1, -3)

        dtype = state_value.dtype
        not_done = 1 - done.to(dtype)
        *batch_size, time_steps = not_done.shape[:-1]

        value = gamma * lmbda
        if isinstance(value, torch.Tensor):
            # create tensor while ensuring that gradients are passed
            gammalmbdas = torch.ones_like(state_value) * not_done * value
        else:
            gammalmbdas = torch.full_like(state_value, value) * not_done
        gammalmbdas = _make_gammas_tensor(gammalmbdas, time_steps, True) # in: [*bs, 100, ts, 1]
        gammalmbdas = gammalmbdas.cumprod(-2)
        # first_below_thr = gammalmbdas < 1e-7
        # # if we have multiple gammas, we only want to truncate if _all_ of
        # # the geometric sequences fall below the threshold
        # first_below_thr = first_below_thr.all(axis=0)
        # if first_below_thr.any():
        #     gammalmbdas = gammalmbdas[..., :first_below_thr, :]

        td0 = reward + not_done * gamma * next_state_value - state_value

        if len(batch_size) > 1:
            td0 = td0.flatten(0, len(batch_size) - 1)
        elif not len(batch_size):
            td0 = td0.unsqueeze(0)

        advantage = _custom_conv1d(td0.transpose(-2, -1), gammalmbdas)

        if len(batch_size) > 1:
            advantage = advantage.unflatten(0, batch_size)
        elif not len(batch_size):
            advantage = advantage.squeeze(0)

        advantage = advantage.transpose(-2, -1)
        value_target = advantage + state_value

        value_target = value_target.transpose(-1, -3).squeeze(-3)
        advantage = advantage.transpose(-1, -3).squeeze(-3)

        return advantage.unsqueeze(-1), value_target
        

