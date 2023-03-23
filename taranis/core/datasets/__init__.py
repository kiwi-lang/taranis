from copy import deepcopy

from torch.utils.data import DataLoader as PytorchDataLoader


class Dataloader:
    def __init__(self, dataset=None, train=None, valid=None, test=None, **kwargs) -> None:
        self.kwargs = kwargs

        if dataset:
            pass

        self.datasets = {
            'train': train,
            'valid': valid,
            'test': test,
        }
        self.loaders = dict()
    
    def _apply_overrides(self, overrides):
        kwargs = deepcopy(self.kwargs)
        kwargs.update(overrides)
        return kwargs
    
    def _new_loader(self, name, dataset, kwargs):
        loader = self.loaders.get(name)
        if loader is None:
            dataset = dataset or self.datasets[name]
            loader = PytorchDataLoader(
                dataset,
                **self._apply_overrides(kwargs),
            )
            self.loaders[name] = loader
        return loader

    def train(self, dataset=None, **overrides):
        return self._new_loader('train', dataset, overrides)

    def validation(self, dataset=None, **overrides):
        if 'shuffle' not in overrides:
            overrides['shuffle'] = False
    
        return self._new_loader('valid', dataset, overrides)

    def test(self, dataset=None, **overrides):
        if 'shuffle' not in overrides:
            overrides['shuffle'] = False
    
        return self._new_loader('test', dataset, overrides)
