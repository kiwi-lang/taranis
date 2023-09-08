import torch
import torch.optim as optim
import torch.nn.functional as F



def generate_image(model, input_shape, label, lr=100, confidence_target=0.99):
    """Extract the internal representation of a label"""
    # Disable model gradient
    model.eval()
    for param in model.parameters():
        param.require_grads = False

    return tweak_image(model, torch.zeros(1, *input_shape), label, lr, confidence_target)


def tweak_image(model, original, label, lr=1, confidence_target=0.99, max_iter=1000):
    # Disable model gradient
    model.eval()
    for param in model.parameters():
        param.require_grads = False

    generated_image = original.cuda().requires_grad_(True)
    assert generated_image.is_leaf

    optimizer = optim.SGD([generated_image], lr=lr)
    confidence = 0

    for _ in range(max_iter):
        optimizer.zero_grad()
        output = model(generated_image)
        loss = F.cross_entropy(output, torch.tensor([label], dtype=torch.long).cuda())
        loss.backward()
        optimizer.step()
        confidence = output[0, label]

        if confidence > confidence_target:
            break

    img = generated_image.detach().cpu()
    return img[0], confidence.item()


def renormalize(data, min, max):
    mn = data.min()
    mx = data.max()
    rg = mx - mn
    normalized = (data - mn) / rg
    return normalized * (max - min) + min