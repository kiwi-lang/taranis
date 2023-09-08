import math

import torch


def show_conv_layers(result):
    """Flatten a tensor (C x W x H) to be displayed as a 2D image"""
    shape = result.shape
    row = math.ceil(math.sqrt(shape[0] + 1))
    layers = torch.zeros(row * shape[1], row * shape[2])
    for r in range(row):
        for c in range(row):
            try:
                sr = r * shape[1]
                sc = c * shape[2]
                layers[sr : sr + shape[1], sc : sc + shape[2]] = result[
                    c * row + r, :, :
                ]
            except:
                break

    return layers


import matplotlib.pyplot as plt

# TESTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

activations = (nn.ReLU,)


def is_perfect_square(v):
    w = int(np.sqrt(v))
    return w * w == v


prev_was_conv = False


def show_layers(layer, input, output):
    global prev_was_conv

    flat = nn.Flatten()(output)
    w = int(np.sqrt(flat.shape[1]))

    if not is_perfect_square(flat.shape[1]):
        w += 1

    if isinstance(layer, nn.Flatten):
        print("Skip flatten")
        return

    print(layer, output.shape)
    if isinstance(layer, nn.Conv2d) or (
        prev_was_conv and isinstance(layer, activations)
    ):
        img = show_conv_layers(output.squeeze(0))
        prev_was_conv = True
    else:
        new = torch.zeros((w * w,), dtype=torch.float)
        new[: flat.shape[1]] = flat[:]
        img = new.view(w, w)
        prev_was_conv = False

    # transforms.ToPILImage()(img).resize((512, 512))
    plt.matshow(img, interpolation=None, cmap="Greys")
    plt.show()


def show_steps(model, dataset):
    with torch.no_grad():
        for i in range(5):
            layer = None
            image, label = dataset[i]

            input = transforms.ToTensor()(image).unsqueeze(0)
            show_layers(layer, input, input)

            for layer in list(model.children()):
                output = layer(input)
                show_layers(layer, input, output)
                input = output
