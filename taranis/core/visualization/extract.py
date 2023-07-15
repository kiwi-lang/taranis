import torch
import torch.optim as optim
import torch.nn.functional as F



def generate_image(model, input_shape, label, lr, confidence_target=0.99):

    # Disable model gradient
    model.eval()
    for param in model.parameters():
        param.require_grads = False


    generated_image = torch.zeros(1, *input_shape).cuda().requires_grad_(True)
    assert generated_image.is_leaf

    optimizer = optim.SGD([generated_image], lr=lr)
    confidence = 0

    while confidence < confidence_target:
        optimizer.zero_grad()
        output = model(generated_image)
        loss = F.cross_entropy(output, torch.tensor([label], dtype=torch.long).cuda())
        loss.backward()
        optimizer.step()
        confidence = output[0, label]

    return generated_image.detach().cpu()[0], confidence