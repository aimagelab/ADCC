import argparse
import os.path as OSPATH
import torchvision.models as models
from metrics import adcc as ADCC
from image_utils import image_utils as IMUT






def main(opt):

    image, saliency_map, explanation_map = IMUT.images_to_tensors(opt)
    arch_name = opt.model.lower()

    arch_dict = {
        'resnet18': models.resnet18(pretrained=True).eval(),
        'resnet50': models.resnet50(pretrained=True).eval(),
        'vgg16': models.vgg16(pretrained=True).eval()
    }
    arch = arch_dict[arch_name]

    return ADCC.ADCC(image, saliency_map, explanation_map, arch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='example/image.png')
    parser.add_argument("--saliency_map", type=str, default='example/salmap.png')
    parser.add_argument("--model", type=str, default='vgg16')

    opt = parser.parse_args()

    assert OSPATH.exists(opt.image), "Image not found"
    assert OSPATH.exists(opt.saliency_map), "Saliency map not found"

    main(opt)

