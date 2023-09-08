from copy import deepcopy

from torch.utils.data import DataLoader as PytorchDataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

import taranis.core.distributed.manager as distributed
from taranis.core.event.observer import Observer


class _EpochSetter(Observer):
    def __init__(self, loader) -> None:
        super().__init__()
        self.loader = loader

    def new_epoch(self, epoch, **kwargs):
        self.loader.set_epoch(epoch)


class Dataloader:
    def __init__(
        self, dataset=None, train=None, valid=None, test=None, **kwargs
    ) -> None:
        self.kwargs = kwargs

        if dataset:
            pass

        self.samplers = {
            "train": dict(shuffle=True),
            "valid": dict(shuffle=False),
            "test": dict(shuffle=False),
        }
        self.datasets = {
            "train": train,
            "valid": valid,
            "test": test,
        }
        self.loaders = dict()

    def set_epoch(self, epoch):
        loader = self.loaders.get("train")
        if loader:
            loader.sampler.set_epoch(epoch)

    def _apply_overrides(self, overrides):
        kwargs = deepcopy(self.kwargs)
        kwargs.update(overrides)
        return kwargs

    def _new_loader(self, name, dataset, kwargs):
        if "sampler" not in kwargs:
            kwargs["sampler"] = self._sampler(name, dataset)

        loader = self.loaders.get(name)
        if loader is None:
            dataset = dataset or self.datasets[name]
            loader = PytorchDataLoader(
                dataset,
                **self._apply_overrides(kwargs),
            )
            self.loaders[name] = loader
        return loader

    def sampler(self, name, **kwargs):
        """Configure the sampler for a given split"""
        self.samplers[name] = kwargs

    def _sampler(self, name, dataset):
        kwargs = self.samplers.get(name, dict())

        if distributed.enabled():
            return DistributedSampler(dataset=dataset, **kwargs)

        return RandomSampler(dataset=dataset, **kwargs)

    def train(self, dataset=None, **overrides):
        return self._new_loader("train", dataset, overrides)

    def validation(self, dataset=None, **overrides):
        return self._new_loader("valid", dataset, overrides)

    def test(self, dataset=None, **overrides):
        return self._new_loader("test", dataset, overrides)
