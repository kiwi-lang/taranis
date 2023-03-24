




class Split:
    def __init__(self, dataset) -> None:
        self.train = None
        self.valid = None
        self.test =  None

    @classmethod
    def from_splits(self, *datasets):
        """Create a merged dataset and split it again"""
        merged = Merged(*datasets)
        return Split(merged)

    def splits(self):
        return self.train, self.valid, self.test


class TransformedDataset:
    """Takes a dataset and transforms it"""
    
    def __init__(self) -> None:
        pass