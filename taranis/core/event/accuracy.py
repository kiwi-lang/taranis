import logging

from .observer import StandardObserver


class Accuracy(StandardObserver):
    def new_epoch(self, **kwargs):
        self.accuracies = []
        self.count = 0

    def end_batch(self, prediction, label, **kwargs):
        n_correct_predictions = prediction.detach().argmax(-1).eq(label).sum()
        self.count += label.shape[0]
        self.accuracies.append(n_correct_predictions)

    def end_epoch(self, **kwargs):
        accuracy = sum([acc.item() for acc in self.accuracies])
        self.save(accuracy=accuracy / self.count)


class OnlineLoss(StandardObserver):
    def new_epoch(self, **kwargs):
        self.losses = []
        self.count = 0

    def end_batch(self, loss, **kwargs):
        self.losses.append(loss.detach())
        self.count += 1

    def end_epoch(self, **kwargs):
        loss = sum([loss.item() for loss in self.losses])
        self.save(online_loss=loss / self.count)


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

    def metrics(self, metrics=None, **kwargs):
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
        loss = self.latest_metrics.get('validation_loss', float('Nan'))
        acc = self.latest_metrics.get('validation_accuracy', float('Nan'))
        self.logger.debug(f"{p} loss {loss:.2f} acc: {acc:.2%}")

    def new_epoch(self, epoch=None, **kwargs):
        self.epoch = epoch

        if not self.latest_metrics:
            self.logger.debug(f"Starting epoch {epoch}/{self.epochs}")
            return
    
        self.show_stat()

    def end_batch(self, loss, **kwargs):
        self.step += 1

        if self.step % self.frequency == 0:
            p = self._progress()
            self.logger.debug(p)
        
    def end_epoch(self, **kwargs):
        self.save(epoch=self.epoch, step=self.step)

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
        from torch import Tensor, nn
        from torch.nn import functional as F

        with torch.no_grad():
            self.model.eval()

            for batch in self.dataloader:
                total_loss = 0.0
                n_samples = 0
                correct_predictions = 0

                for batch in self.dataloader:
                    batch = tuple(item.to(self.device) for item in batch)
                    x, y = batch

                    logits: Tensor = self.model(x)
                    loss = F.cross_entropy(logits, y)

                    batch_n_samples = x.shape[0]
                    batch_correct_predictions = logits.argmax(-1).eq(y).sum()

                    total_loss += loss.item()
                    n_samples += batch_n_samples
                    correct_predictions += batch_correct_predictions

            accuracy = correct_predictions / n_samples
            self.save(validation_loss=total_loss, validation_accuracy=accuracy.item())
