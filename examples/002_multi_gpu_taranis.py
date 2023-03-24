"""Multi-GPU Training example."""
import logging
import os

import rich.logging
import torch
import torch.distributed
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from taranis.core.dataset import Dataloader
from taranis.core.event.manager import EventManager, Saver
from taranis.core.event.accuracy import OnlineAccuracy, OnlineLoss, Progress, Validation
from taranis.core.device import cpu_count
import taranis.core.distributed.manager as distributed 


def main():
    training_epochs = 10
    learning_rate = 5e-4
    weight_decay = 1e-4
    batch_size = 128  # NOTE: This is the "local" batch size, per-GPU.

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0

    rank = distributed.rank()
    world_size =  distributed.world_size()
    is_master = distributed.has_weight_autority()
    device = distributed.device()

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
    )

    logger = logging.getLogger(__name__)

    # Create a model and move it to the GPU.
    # use local_model for checkpointing
    local_model = resnet18(num_classes=10)
    model = distributed.dataparallel(local_model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # Setup CIFAR10
    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")
    train_dataset, valid_dataset, test_dataset = make_datasets(dataset_path, is_master=is_master)

    loader = Dataloader(
        train=train_dataset,
        valid=valid_dataset,
        test=test_dataset,
        batch_size=batch_size,
        num_workers=min(cpu_count(), 1),
    )

    # NOTE: Here `batch_size` is still the "local" (per-gpu) batch size.
    # This way, the effective batch size scales directly with number of GPUs, no need to specify it
    # in advance. You might want to adjust the learning rate and other hyper-parameters though.
    if is_master:
        logger.info(f"Effective batch size: {batch_size * world_size}")
    
    train_dataloader = loader.train(train_dataset)
    valid_dataloader = loader.validation(valid_dataset)
    test_dataloader = loader.test(test_dataset)
    
    events = EventManager(Saver())
    events.register(OnlineAccuracy())
    events.register(OnlineLoss())
    events.register(Validation(loader.validation(), model, device))
    events.register(Progress(training_epochs, len(loader.train()), frequency=1))

    events.new_train()
    for epoch in range(training_epochs):
        events.new_epoch(epoch=epoch)

        # NOTE: Here we need to call `set_epoch` so the ordering is able to change at each epoch.
        loader.set_epoch(epoch)

        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
        model.train()

        # Training loop
        for i, batch in enumerate(train_dataloader):
            events.new_batch(batch_id=i, batch=batch)

            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            # Forward pass
            logits: Tensor = model(x)

            local_loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            local_loss.backward()
            # NOTE: nn.DistributedDataParallel automatically averages the gradients across devices.
            optimizer.step()

            events.end_batch(loss=local_loss, prediction=logits, label=y)
        events.end_epoch(epoch=epoch)
    events.end_train()
    print("Done!")


def make_datasets(
    dataset_path: str,
    is_master: bool,
    val_split: float = 0.1,
    val_split_seed: int = 42,
):
    """Returns the training, validation, and test splits for CIFAR10.

    NOTE: We don't use image transforms here for simplicity.
    Having different transformations for train and validation would complicate things a bit.
    Later examples will show how to do the train/val/test split properly when using transforms.

    NOTE: Only the master process (rank-0) downloads the dataset if necessary.
    """
    # - Master: Download (if necessary) THEN Barrier
    # - others: Barrier THEN *NO* Download
    if not distributed.has_dataset_autority():
        torch.distributed.barrier()

    train_dataset = CIFAR10(
        root=dataset_path, 
        transform=transforms.ToTensor(), 
        download=distributed.has_dataset_autority(),
        train=True
    )
    test_dataset = CIFAR10(
        root=dataset_path, 
        transform=transforms.ToTensor(), 
        download=distributed.has_dataset_autority(), 
        train=False
    )

    # Join the workers waiting in the barrier above. They can now load the datasets from disk.
    if is_master:
        torch.distributed.barrier()

    # Split the training dataset into a training and validation set.
    train_dataset, valid_dataset = random_split(
        train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
    )
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    with distributed.distributed(enabled=True):
        main()
