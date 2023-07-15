import torch.optim as optim
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import TensorDataset

from ..dataset.split import TransformedDatasetClassification


def cifar100_dataset(train=True):
    transform = transforms.Compose([                    
        transforms.ToTensor(),                         # Transform the image to tensor
        transforms.Normalize(
            mean=[n/255. for n in [129.3, 124.1, 112.4]], 
            std=[n/255. for n in [68.2,  65.4,  70.4]]
        )
    ])

    dataset = datasets.CIFAR100(
        '../data', 
        train=train, 
        download=True
    )

    sizes = transform(dataset[0][0]).shape
    inputs = torch.zeros(len(dataset), *sizes, dtype=torch.float)
    outputs = torch.zeros(len(dataset), dtype=torch.long)

    for i, (x, y) in enumerate(dataset):
        inputs[i] = transform(x)
        outputs[i] = y

    return TensorDataset(inputs.cuda(), outputs.cuda())


def mnist_dataset(train=True):
    transform = transforms.Compose([                    
        transforms.ToTensor(),                         # Transform the image to tensor
        transforms.Normalize((0.1307,), (0.3081,)),    # Normalize the image
    ])

    dataset = datasets.MNIST(
        '../data', 
        train=train, 
        download=True
    )

    sizes = transform(dataset[0][0]).shape
    inputs = torch.zeros(len(dataset), *sizes, dtype=torch.float)
    outputs = torch.zeros(len(dataset), dtype=torch.long)

    for i, (x, y) in enumerate(dataset):
        inputs[i] = transform(x)
        outputs[i] = y

    return TensorDataset(inputs.cuda(), outputs.cuda())


def gpu_train(model, dataset, batch_size=4096, lr=1):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 0,
    )

    assert torch.cuda.is_available()
    # device = torch.cuda.device("cuda:0")               # HERE Get GPU device

    cuda_model = model.cuda()                # HERE
    optimizer = optim.Adadelta(cuda_model.parameters(), lr=lr)
    epoch = 10

    for i in range(epoch):
        losses = []
        count = 0
        for batch, labels in dataloader:
            batch, labels = batch.cuda(), labels.cuda() # HERE

            optimizer.zero_grad()
            probabilities = cuda_model(batch)
            loss = F.nll_loss(probabilities, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())
            count += 1
        
        total_loss = (sum(losses) / count).item()
        print(i, total_loss / count)
    

def gpu_test_model(model, dataset, batch_size=4096):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0) 
    total = len(dataset)
    cuda_model = model.cuda()    

    with torch.no_grad():
        test_loss = 0
        correct = 0
        
        for batch, labels in loader:
            batch, labels = batch.cuda(), labels.cuda()
            
            output = cuda_model(batch)
            test_loss += F.nll_loss(output, labels, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(labels.view_as(pred)).sum().item()

        print('Accuracy:', correct / total * 100, 'Loss:', test_loss / total)