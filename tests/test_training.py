from data.default_loader import FaceDsLoader
from model.autoenc_deconv import Autoenc_deconv
from training.trainer import FaceSwapTrainer
from .utils import read_default_config

def test_training_makes_one_pass_without_error():
    for ds_path_p1, ds_path_p2, has_validation in [("__dataset/serena/", "__dataset/novak/", False), ("__dataset2_13/serena/", "__dataset2_13/novak/", True), ("__dataset2_13-rev1/serena/", "__dataset2_13-rev1/novak/", True)]:
        cfg = read_default_config()
        data_loader = FaceDsLoader(ds_path_p1, ds_path_p2, None if not has_validation else {'has_validation': True})
        model = Autoenc_deconv()
        trainer = FaceSwapTrainer(cfg, data_loader=data_loader, model=model)
    
        trainer.train_step()