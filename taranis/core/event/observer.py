


class Observer:
    _events = None
    _manager = None

    def __init__(self) -> None:
        self._events = []

        for fun in dir(self):
            if fun.startswith('_') or fun in ('save',):
                continue
        
            self._events.append(fun)

    def save(self, **kwargs):
        self._manager.save(**kwargs)


class StandardObserver(Observer):

    def new_train(self, **kwargs):
        pass

    def new_epoch(self, **kwargs):
        pass

    def new_batch(self, **kwargs):
        pass

    def end_batch(self, **kwargs):
        pass

    def end_epoch(self, **kwargs):
        pass

    def end_train(self, **kwargs):
        pass


StandardObserver()


