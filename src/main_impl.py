from typing import Any
from omegaconf import DictConfig, OmegaConf

def main_impl(cfg: DictConfig, secrets: Any = None):
    if cfg.preview_dataset:
        import data.default_loader as data
        import matplotlib.pyplot as plt
        import data.default_loader as default_loader
        ds_path_p1 = cfg.prev_ds_path_p1
        ds_path_p2 = cfg.prev_ds_path_p2
        if ds_path_p1 is None or ds_path_p2 is None:
            data_loader = data.FaceDsLoader()
        else:
            data_loader = data.FaceDsLoader(ds_path_p1, ds_path_p2)
        #force batch_size 1,        #TODOcfg.training.batch_size = 1
        cfg.training.val_batch_size = 1
        cfg.training.test_batch_size = 1
        cfg.training.val_split = 0.2
        ds = data_loader(cfg)
        preprocess = default_loader.get_preprocessing_layers(data_loader.ds_mean)
        r=1
        for x,y in ds.x_p1.take(5):
            plt.subplot(5,3,r)
            r += 1
            plt.imshow(x[0]/255.0)
            plt.subplot(5,3,r)
            r += 1
            plt.imshow(preprocess(x)[0])
            plt.subplot(5,3,r)
            r += 1
            plt.imshow(y[0])
        plt.show()
    elif cfg.runner.name == "ray":
        import runner.ray_runner as ray_trainer
        ray_trainer.start_tune_training(cfg)