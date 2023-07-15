




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



def _no_transforms(x):
    return x


class TransformedDataset:
    """Takes a dataset and transforms it"""
    
    def __init__(self, dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform or _no_transforms

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])
    
    def __len__(self):
        return len(self.dataset)


class TransformedDatasetClassification:
    def __init__(self, dataset, transform, target_transform = None) -> None:
        self.dataset = dataset
        self.transform = transform  or _no_transforms
        self.target_transform = target_transform or _no_transforms

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return self.transform(data), self.target_transform(target)
    
    def __len__(self):
        return len(self.dataset)