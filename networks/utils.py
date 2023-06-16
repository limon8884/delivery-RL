import torch

def create_mask(lenghts, device)->torch.BoolTensor:
    '''
    returns True if element is masked, False otherwise
    '''
    max_len = max(lenghts)
    with torch.no_grad():
        if max_len == 0:
            return torch.ones((len(lenghts), 1), device=device, dtype=torch.bool)
        result = torch.arange(max_len, device=device).expand(len(lenghts), max_len) >= torch.tensor(lenghts, device=device).unsqueeze(1)
        return result
    