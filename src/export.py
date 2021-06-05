import tensorflow as tf
import sys
import os
import json
from config_utils import read_experiment_params
from model.impl_factory import model_impl_factory as get_autoenc_impl_from_config
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def main(artifacts_dir_path, export_dir, checkpoint_dir, custom_name = None):
    params = read_experiment_params(artifacts_dir_path)
    print("loading model " + params['model']['name'])
    model = get_autoenc_impl_from_config(params, ds_mean=[0,0,0])
    model.build((1,96,96,3))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.read(os.path.join(artifacts_dir_path, checkpoint_dir, "model")).assert_consumed()
    model.summary()
    
    func = tf.function(lambda x: model(x))
    func = func.get_concrete_function(tf.TensorSpec((1,96,96,3), tf.float32))
    frozen_func = convert_variables_to_constants_v2(func)
    frozen_func.graph.as_graph_def()

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    existing_count = len(os.listdir(export_dir))
    if custom_name == None:
        file_name = "model.pb" if existing_count == 0 else f"model{existing_count+1}.pb"
    else:
        file_name = custom_name

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, 
                    logdir=export_dir, name=file_name, as_text=False)
                    
    #tf.io.write_graph(graph_or_graph_def=frozen_func.graph, 
    #                logdir=export_dir, name="model.pbtxt", as_text=True)


if __name__ == '__main__':
    print(sys.argv)
    #D:\\WIN10_SSDSamsung\\TEMP\\jupyter-outputs\\gradient_arch_batch_dataset2_best_perf\\jupyter-outputs\\ray_results\\arch_batch_dataset2\\TuneTrainer_6cba2_00000
    art_dir = "D:\\WIN10_SSDSamsung\\Documents\\FaceSwapResults\\ds3exp1\\ds3exp1_batch_norm_masked_regularized\\jupyter-outputs\\ray_results\\ds3_exp1\\TuneTrainer_70733_00000"
    save_dir = "./__saves__"
    cdir = "checkpoint_000045"
    custom_name = "model_masked1.pb"
    main(art_dir,save_dir,cdir, custom_name)