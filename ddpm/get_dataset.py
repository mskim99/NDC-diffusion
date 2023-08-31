import sys
sys.path.append('../')
from ddpm.default import DEFAULTDataset


def get_dataset(cfg):
    train_dataset = DEFAULTDataset(
        root_dir=cfg.dataset.root_dir)
    val_dataset = DEFAULTDataset(
        root_dir=cfg.dataset.root_dir)
    sampler = None
    return train_dataset, val_dataset, sampler
