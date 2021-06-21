from metrics import coherency, complexity, average_drop

def ADCC(image, saliency_map, explanation_map,arch,attr_method,target_class_idx=None):

    avgdrop = average_drop.average_drop(image, explanation_map, arch, class_idx=target_class_idx)
    coh=coherency.coherency(saliency_map,explanation_map,attr_method, class_idx=target_class_idx)
    com=complexity.complexity(saliency_map)


    adcc = 3 * (1/coh + 1/(1-com) +1/(1-avgdrop))

    return adcc