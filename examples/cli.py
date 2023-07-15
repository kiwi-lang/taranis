import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import taranis.core.dataset.split as split
import torch.optim as optim
import torch.nn.functional as F


def gpu_train(original_model, lr, epoch=10, batch_size=4096):

    dataset = datasets.MNIST(
        '../data', 
        train=True, 
        download=True
    )
    print(len(dataset))


    device = torch.cuda.current_device()               # HERE Get GPU device
    model = original_model.to(device)                  # HERE
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    dataloader = torch.utils.data.DataLoader(
        split.TransformedDatasetClassification(
            dataset, 
            transform=transforms.Compose([                    
                transforms.ToTensor(),                         # Transform the image to tensor
                transforms.Normalize((0.1307,), (0.3081,)),    # Normalize the image
            ])
        ),
        batch_size = batch_size,
        num_workers = 1,
    )


    for i in range(epoch):
        losses = []
        count = 0
        for batch, labels in dataloader:
            batch, labels = batch.to(device), labels.to(device) # HERE

            optimizer.zero_grad()
            probabilities = model(batch)
            loss = F.nll_loss(probabilities, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())
            count += 1
        total_loss = (sum(losses) / count).item()
        print(i, total_loss / count)
    
def gpu_test_model(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=False,  transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=1) 
    total = len(dataset)
    device = torch.cuda.device("cuda:0")  # HERE
    model = model.to(device=device)       # HERE
    with torch.no_grad():
        test_loss = 0
        correct = 0
        
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device) # HERE
            
            output = model(batch)
            test_loss += F.nll_loss(output, labels, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(labels.view_as(pred)).sum().item()

        print('Accuracy', correct / total * 100)


if __name__ == '__main__':
    conv_model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 576, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1),
    )

    gpu_train(conv_model, epoch=100, lr=0.5, batch_size=4092)
    gpu_test_model(conv_model)