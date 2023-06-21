import torch.utils.data
from data.base_dataset import collate_fn
from data.segmentation_data import SegmentationData

def CreateDataset(opt):
    """loads dataset class"""

    dataset = SegmentationData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn,
            drop_last = True)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data