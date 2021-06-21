import torch
import PIL.Image as IMAGE
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

def apply_transform(image, size=224, means= torch.tensor([0.485, 0.456, 0.406]), stds=torch.tensor([0.229, 0.224, 0.225])):

    if not isinstance(image, IMAGE.Image):
        image = F.to_pil_image(image)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def detransform(tensor):
    means, stds = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    denormalized = transforms.Normalize(-1 * means / stds, 1.0 / stds)(tensor)

    return denormalized


def load_image(image_path):
    return IMAGE.open(image_path).convert('RGB')

def images_to_tensors(opt):
    image=load_image(opt.image)
    saliency_map=load_image(opt.saliency_map)

    image=apply_transform(image)
    saliency_map=apply_transform(saliency_map)

    return image, saliency_map, image * saliency_map
