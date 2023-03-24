"""Single-GPU training example."""
import logging
import os

import rich.logging
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR100, FakeData
from torchvision.models import resnet18

from taranis.core.dataset import Dataloader
from taranis.core.event.manager import EventManager, Saver
from taranis.core.event.accuracy import OnlineAccuracy, OnlineLoss, Progress, Validation
from taranis.core.device import cpu_count


def main():
    training_epochs = 10
    learning_rate = 5e-4
    weight_decay = 1e-4
    batch_size = 128

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.DEBUG,
        # Very pretty, uses the `rich` package.
        handlers=[rich.logging.RichHandler(markup=True)],  
    )

    # Create a model and move it to the GPU.
    model = resnet18(num_classes=100)
    model = model.to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
    )

    # Setup CIFAR10
    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")
    train_dataset, valid_dataset, test_dataset = make_datasets(dataset_path)

    loader = Dataloader(
        train=train_dataset,
        valid=valid_dataset,
        test=test_dataset,
        batch_size=batch_size,
        num_workers=min(cpu_count(), 1),
    )

    events = EventManager(Saver())
    events.register(OnlineAccuracy())
    events.register(OnlineLoss())
    events.register(Validation(loader.validation(), model, device))
    events.register(Progress(training_epochs, len(loader.train()), frequency=1))
    
    events.new_train()
    for epoch in range(training_epochs):
        events.new_epoch(epoch=epoch)

        # Set the model in training mode (this is important for e.g. BatchNorm and Dropout layers)
        model.train()

        # Training loop
        for i, batch in enumerate(loader.train()):
            events.new_batch(batch_id=i, batch=batch)

            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            # Forward pass
            logits: Tensor = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---
            events.end_batch(loss=loss, prediction=logits, label=y)
        events.end_epoch(epoch=epoch)
    events.end_train()
    print("Done!")


def make_datasets(
    dataset_path: str,
    val_split: float = 0.1,
    val_split_seed: int = 42,
):
    """Returns the training, validation, and test splits for CIFAR10.

    NOTE: We don't use image transforms here for simplicity.
    Having different transformations for train and validation would complicate things a bit.
    Later examples will show how to do the train/val/test split properly when using transforms.
    """
    train_dataset = FakeData(
        image_size=(3, 32, 32),
        # root=dataset_path, 
        transform=transforms.ToTensor(), 
        # download=True, 
        # train=True
    )
    test_dataset = FakeData(
        size=100,
        image_size=(3, 32, 32),
        # root=dataset_path, 
        transform=transforms.ToTensor(), 
        # download=True, 
        # train=False
    )
    # Split the training dataset into a training and validation set.
    train_dataset, valid_dataset = random_split(
        train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
    )
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    main()
