import torch
import torch.nn as nn

class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, state):
        # <Implement policy by calling model, sampling actions and computing their log probs>
        # FIX handle inputs
        actor, critic = self.model(state)
    
        # probs = nn.functional.softmax(a, dim=-1).detach().numpy()
        # actions = np.array([np.random.choice(6, p=probs[i]) for i in range(len(inputs))])
        action_probs = nn.functional.softmax(actor, dim=-1)
        action = torch.distributions.Categorical(action_probs).sample()
        # logits = torch.gather(a, dim=1, index=torch.tensor(actions).unsqueeze(-1)).squeeze()

        log_softmaxed = nn.functional.log_softmax(actor, dim=-1)

        # print(log_softmaxed[action.unsqueeze(-1)])
        log_probs = torch.gather(log_softmaxed, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)

        d = {}
        d['actions'] = action
        d['logits'] = actor
        d['log_probs'] = log_probs
        d['values'] = critic

        return d
    
        # Should return a dict containing keys ['actions', 'logits', 'log_probs', 'values'].