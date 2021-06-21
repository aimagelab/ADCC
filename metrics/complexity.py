from scipy.linalg import norm

def complexity(saliency_map):
    return abs(saliency_map).sum()/(saliency_map.shape[-1]*saliency_map.shape[-2])