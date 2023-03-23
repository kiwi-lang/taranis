from collections import defaultdict
from typing import List
from functools import partial
from copy import deepcopy

from .observer import Observer


class Saver:
    def __init__(self) -> None:
        pass

    def metrics(self, **kwargs):
        pass


class EventManager:
    def __init__(self, saver=None) -> None:
        self.observers: List[Observer] = []
        self.handlers = defaultdict(list)
        self.saver = saver
        
    def emit_event(self, name, **kwargs):
        self.metrics = dict()

        for handler in self.handlers[name]:
            getattr(handler, name)(**kwargs)

        if self.metrics:
            assert name != 'metrics', 'Cannot save metrics, modify the dictionary directly'
            metrics = deepcopy(self.metrics)
            self.emit_event('metrics', metrics=metrics)

            if self.saver:
                self.saver.metrics(**metrics)
        
        self.metrics = dict()

    def save(self, **kwargs):
        self.metrics.update(kwargs)

    def register(self, observer):
        self.observers.append(observer)
        observer._manager = self
        
        for event in observer._events:
            self.handlers[event].append(observer)

    def __getattr__(self, __name: str) :
        return partial(self.emit_event, __name)
