from scipy.linalg import norm

def complexity(saliency_map):
    return norm(saliency_map, ord=1)