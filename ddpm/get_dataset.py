import sys
sys.path.append('../')
from ddpm.default import DEFAULTDataset


def get_dataset(cfg):
    train_dataset = DEFAULTDataset(
        root_dir='/data/jionkim/gt_NDC_KISTI_SDF_p_100_npy/res_64_int/')
    val_dataset = DEFAULTDataset(
        root_dir='/data/jionkim/gt_NDC_KISTI_SDF_p_100_npy/res_64_int/')
    sampler = None
    return train_dataset, val_dataset, sampler
