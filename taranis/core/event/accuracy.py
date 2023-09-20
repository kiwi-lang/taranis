import logging
from datetime import datetime

import taranis.core.distributed.manager as distributed

from ..device import sync
from .observer import StandardObserver


class OnlineAccuracy(StandardObserver):
    def new_epoch(self, **kwargs):
        self.accuracies = []
        self.count = 0

    def end_batch(self, prediction, label, **kwargs):
        n_correct_predictions = prediction.detach().argmax(-1).eq(label).sum()
        self.count += label.shape[0]
        self.accuracies.append(n_correct_predictions)

    def end_epoch(self, **kwargs):
        accuracy = sum(acc.item() for acc in self.accuracies)
        self.save(accuracy=accuracy / self.count)


class OnlineLoss(StandardObserver):
    def new_epoch(self, **kwargs):
        self.losses = []
        self.count = 0

    def end_batch(self, loss, **kwargs):
        self.losses.append(loss.detach())
        self.count += 1

    def end_epoch(self, **kwargs):
        loss = sum(loss.item() for loss in self.losses)
        self.save(online_loss=loss / self.count)


class DistributedObserver(StandardObserver):
    def __init__(self) -> None:
        super().__init__()
        self.rank = distributed.grank()
        self.world_size = distributed.world_size()
        self.logger = logging.getLogger("distributed")

    def metrics(self, metrics=None, **kwargs):
        if self.rank >= 0 and metrics:
            metrics["rank"] = self.rank
            metrics["world_size"] = self.world_size

    def new_train(self, **kwargs):
        if distributed.has_metric_autority():
            self.logger(f"World size: {self.world_size}, global rank: {self.rank}")


class Progress(StandardObserver):
    def __init__(self, epochs, batches, frequency=100) -> None:
        super().__init__()
        self.epochs = epochs
        self.batches = batches
        self.total = self.epochs * self.batches
        self.frequency = frequency
        self.step = 0
        self.logger = logging.getLogger("Progress")
        self.latest_metrics = dict()
        self.epoch = 0
        self.ended = False
        self.previous = None

    def metrics(self, metrics=None, **kwargs):
        metrics["epoch"] = self.epoch
        metrics["step"] = self.step

        self.latest_metrics.update(metrics)

        if self.ended:
            self.show_stat()

    def new_train(self, **kwargs):
        # Checkout the "checkpointing and preemption" example for more info!
        self.logger.debug("Starting training from scratch.")

    def _progress(self):
        return f"[{self.epoch}/{self.epochs}][{self.step}/{self.total}]"

    def show_stat(self):
        p = self._progress()
        loss = self.latest_metrics.get("validation_loss", float("Nan"))
        acc = self.latest_metrics.get("validation_accuracy", float("Nan"))
        self.logger.debug(f"{p} loss {loss:.2f} acc: {acc:.2%}")

    def new_epoch(self, epoch=None, **kwargs):
        self.epoch = epoch

        if not self.latest_metrics:
            self.logger.debug(f"Starting epoch {epoch}/{self.epochs}")
            return

        self.track_epoch_time()
        self.show_stat()

    def track_epoch_time(self):
        # Synchronise to make sure epoch times are accurate
        sync()

        now = datetime.utcnow()
        kwargs = {}
        kwargs["datetime"] = now

        if self.previous:
            kwargs["elapsed"] = (now - self.previous).total_seconds()

        self.previous = now
        self.save(**kwargs)

    def end_batch(self, loss, **kwargs):
        self.step += 1

        if self.step % self.frequency == 0:
            p = self._progress()
            self.logger.debug(p)

    def end_train(self, **kwargs):
        self.epoch += 1
        self.ended = True


class Validation(StandardObserver):
    def __init__(self, dataloader, model, device) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.model = model
        self.device = device

    def end_epoch(self, **kwargs):
        self.compute_validation()

    def end_train(self, **kwargs):
        # self.compute_validation()
        pass

    def compute_validation(self):
        import torch
        from torch import Tensor
        from torch.nn import functional as F

        with torch.no_grad():
            self.model.eval()

            for batch in self.dataloader:
                total_losses = []
                n_samples = 0
                correct_predictions = 0

                for batch in self.dataloader:
                    batch = tuple(item.to(self.device) for item in batch)
                    x, y = batch

                    logits: Tensor = self.model(x)
                    loss = F.cross_entropy(logits, y)

                    batch_n_samples = x.shape[0]
                    batch_correct_predictions = logits.argmax(-1).eq(y).sum()

                    total_losses.append(loss.detach())
                    n_samples += batch_n_samples
                    correct_predictions += batch_correct_predictions

            # Local
            total_loss = sum(total_losses)
            if not distributed.has_dataset_autority():
                self.save(
                    local_validation_loss=total_loss.items(),
                    local_validation_accuracy=correct_predictions.item() / n_samples,
                )

            else:
                rank = distributed.grank()

                distributed.reduce(total_loss, dst=rank)
                distributed.reduce(
                    torch.as_tensor(n_samples, device=self.device), dst=rank
                )
                distributed.reduce(correct_predictions, dst=rank)

                total_loss = total_loss / distributed.world_size()
                accuracy = correct_predictions / n_samples

                self.save(
                    validation_loss=total_loss.items(),
                    validation_accuracy=accuracy.item(),
                )


