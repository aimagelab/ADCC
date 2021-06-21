import torch
import torch.nn.functional as FF

def average_drop(image, explanation_map, arch, class_idx=None):
    if torch.cuda.is_available():
        inp = image.cuda()
        expmap=explanation_map.cuda()
        arch = arch.cuda()


    with torch.no_grad():
        out_on_inp = FF.softmax(arch(inp), dim=1)
        out_on_exp = FF.softmax(arch(expmap), dim=1)

    confidence_on_inp = out_on_inp.max(1)[0].item()

    if class_idx is None:
        class_idx = out_on_inp.max(1)[1].item()

    confidence_on_exp = out_on_exp[:,class_idx][0].item()

    return max(0.,confidence_on_inp-confidence_on_exp)/confidence_on_inp