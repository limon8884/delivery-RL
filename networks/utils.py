import torch

def create_mask(lenghts, device):
    max_len = max(lenghts)
    with torch.no_grad():
        result = torch.arange(max_len, device=device).expand(len(lenghts), max_len) >= torch.tensor(lenghts, device=device).unsqueeze(1)
        return result
    