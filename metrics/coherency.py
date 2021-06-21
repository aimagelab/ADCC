from scipy import stats as STS
import torch

def coherency(saliency_map, explanation_map, arch, attr_method, out):
    if torch.cuda.is_available():
        explanation_map = explanation_map.cuda()
        arch = arch.cuda()

    class_idx = out.max(1)[1].item()
    saliency_map_B=attr_method(image=explanation_map, model=arch, classidx=class_idx)

    A, B = saliency_map.detach(), saliency_map_B.detach()

    '''
    # Pearson correlation coefficient
    # '''
    Asq, Bsq = A.view(1, -1).squeeze(0).cpu(), B.view(1, -1).squeeze(0).cpu()
    y, _ = STS.pearsonr(Asq, Bsq)
    y = abs(y)

    return y