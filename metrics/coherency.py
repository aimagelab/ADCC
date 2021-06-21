from scipy import stats as STS

def coherency(saliency_map, explanation_map, arch, attr_method, class_idx=None):

    saliency_map_B=attr_method(explanation_map,arch,class_idx)

    A, B = saliency_map.detach(), saliency_map_B.detach()

    '''
    # Pearson correlation coefficient
    # '''
    Asq, Bsq = A.view(1, -1).squeeze(0).cpu(), B.view(1, -1).squeeze(0).cpu()
    y, _ = STS.pearsonr(Asq, Bsq)
    y = abs(y)

    return y