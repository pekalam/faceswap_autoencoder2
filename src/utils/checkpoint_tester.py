import sys
import os
import tensorflow as tf
from .config_utils import read_experiment_params
from model.impl_factory import model_impl_factory



def check_is_checkpoint_loading(artifacts_dir: str, checkpoint_dir):
    params = read_experiment_params(artifacts_dir)
    model = model_impl_factory(params)
    #TODO change shape
    model.build((1,96,96,3))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.read(os.path.join(artifacts_dir, checkpoint_dir, "model")).assert_consumed()
    print("checkpoint successfully loaded")



if __name__ == '__main__':
    artifacts_dir = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    check_is_checkpoint_loading(artifacts_dir, checkpoint_dir)