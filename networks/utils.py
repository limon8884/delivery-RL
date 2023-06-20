import torch

def create_mask(lenghts, device, mask_first)->torch.BoolTensor:
    '''
    returns True if element is masked, False otherwise
    '''
    max_len = max(lenghts)
    with torch.no_grad():
        assert max_len > 0, 'Tensor should not be empty'
        result = torch.arange(max_len, device=device).expand(len(lenghts), max_len) >= torch.tensor(lenghts, device=device).unsqueeze(1)
        if mask_first:
            result[:, 0] = True
        return result
    