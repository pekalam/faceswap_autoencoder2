import ray
from data.default_loader import FaceDsLoader
from training.trainer import FaceSwapTrainer

from ray.tune.resources import Resources

from ray.tune.logger import TBXLoggerCallback
import os
from ray import tune
import tensorflow as tf
from ray.tune.integration.mlflow import MLflowLoggerCallback
from omegaconf import DictConfig, OmegaConf
import ray.tune
from hydra.utils import instantiate



class CustomTBXLoggerCallback(TBXLoggerCallback):
    def log_trial_result(self, iteration: int, trial, result: dict):
        result = {k:result[k] for k in result.keys() if k not in('image', 'image_p1', 'image_p2', 'should_checkpoint', 'checkpoint', 'iterations_since_restore', 'timesteps_total','time_since_restore','timesteps_since_restore','done','mean','p1_seed', 'p2_seed','stop')}
        return super().log_trial_result(iteration, trial, result)


class CustomMLflowLoggerCallback(MLflowLoggerCallback):
    def __init__(self, tracking_uri=None, registry_uri=None, experiment_name=None, save_artifact: bool = False):
        self.artifact_sent = False
        self.saved_files = []
        super().__init__(tracking_uri=tracking_uri, registry_uri=registry_uri, experiment_name=experiment_name, save_artifact=save_artifact)

    def log_trial_result(self, iteration: int, trial, result: dict):
        run_id = self._trial_runs[trial]

        if result.get('image', None) is not None:
            self.client.log_image(run_id, result.get('image'), "test.png")
        if result.get('mean', None) is not None:
            self.client.log_param(run_id, 'ds_mean', result.get('mean'))
        if result.get('seed', None) is not None:
            self.client.log_param(run_id, 'seed', result.get('seed')) 
        super().log_trial_result(iteration, trial, result)

    def log_trial_end(self, trial, failed: bool):
        self.artifact_sent = True
        self.saved_files = [os.path.join(trial.logdir, f) for f in os.listdir(trial.logdir)]
        return super().log_trial_end(trial, failed=failed)
    
    def log_trial_save(self, trial):
        #last checkpoint not saved - fix
        if self.artifact_sent and self.save_artifact:
            diff = list(set([os.path.join(trial.logdir, f) for f in os.listdir(trial.logdir)]) - set(self.saved_files))
            if len(diff) == 0:
                return
            run_id = self._trial_runs[trial]
            for path in diff:
                self.client.log_artifact(run_id, local_path=path)
        return super().log_trial_save(trial)
    
    def on_trial_save(self, iteration: int, trials, trial, **info):
        return super().on_trial_save(iteration, trials, trial, **info)
    
    def on_trial_complete(self, iteration: int, trials, trial, **info):
        return super().on_trial_complete(iteration, trials, trial, **info)


class TuneTrainer(tune.Trainable):
    def setup(self, config: dict):
        self.data_loader = config['data_loader']
        self.trainer = FaceSwapTrainer(config, data_loader=self.data_loader, preview_logdir=self._logdir)
        self.ckpt = tf.train.Checkpoint(model=self.trainer.model)



    @classmethod
    def default_resource_request(cls, config):
        if config.get('cpus') is not None and config.get('gpus') is not None:
            return Resources(
                config['cpus'],
                config['gpus'])
        else:
            return None


    def save_checkpoint(self, tmp_checkpoint_dir):
        self.ckpt.write(os.path.join(tmp_checkpoint_dir, "model"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir: str):
        print("loading checkpoint from: ", tmp_checkpoint_dir)
        print("checkpoint dir content: ", os.listdir(tmp_checkpoint_dir))
        if self._iteration is not None:
            self.trainer.step = tf.Variable(self._iteration, dtype=tf.int64)
        self.ckpt.read(os.path.join(tmp_checkpoint_dir, "model")).assert_consumed()

    def step(self):
        result = self.trainer.train_step()
        grads_p1 = result.pop('grads_p1')
        grads_p2 = result.pop('grads_p2')

        if result.get('image', None) is not None:
            result.pop('image')
        if result.get('image_p1', None) is not None:
            result.pop('image_p1')
        if result.get('image_p2', None) is not None:
            result.pop('image_p2')
        
        if self.config['training']['show_gradients']:
            result = {**result, **grads_p1, **grads_p2}

        if result.get('stop') == True:
            result[tune.result.DONE] = True
            result.pop('stop')

        if result.get('checkpoint', None) is not None and result.get('checkpoint') == True:
            result[tune.result.SHOULD_CHECKPOINT] = True
            result.pop('checkpoint')

        return result





def start_tune_training(cfg: DictConfig):
    ray.init(**cfg.runner.init_config)
    data_loader = FaceDsLoader(cfg.dataset.ds_path_p1, cfg.dataset.ds_path_p2, cfg.dataset)

    tune_args = {}
    tune_args['callbacks'] = [
            CustomTBXLoggerCallback(),
            
    ]

    if cfg.runner.include_mlflow == True:
        tune_args['callbacks'].append(
            CustomMLflowLoggerCallback(
                tracking_uri=cfg.runner.mlflow.tracking_uri,
                experiment_name=cfg.runner.mlflow.experiment_name,
                save_artifact=cfg.runner.mlflow.save_artifact
            )
        )

    tune_args = {**tune_args, **cfg.runner.run}

    if tune_args.get('search_alg') is not None:
        tune_args['search_alg'] = instantiate(tune_args['search_alg'])

    if cfg.runner.single_seed == True:
        if cfg.training.seed is None:
            cfg.training.seed = int(time.time())
            print('Generated single tune seed: ', cfg.training.seed)
        else:
            print('Using existing single tune seed: ', cfg.training.seed)

    modelhp = {}
    trainhp = {}
    if 'tune' in cfg.runner:
        if 'model' in cfg.runner.tune:
            print('Configuring model hyperparameters')
            for k in cfg.runner.tune.model.keys():
                modelhp[k] = instantiate(cfg.runner.tune.model[k])
                if modelhp[k].get('grid_search', None) is not None:
                    modelhp[k]['grid_search'] = list(modelhp[k]['grid_search'])
            print('Configured modelhp:', modelhp)
        if 'training' in cfg.runner.tune:
            print('Configuring training hyperparameters')
            for k in cfg.runner.tune.training.keys():
                trainhp[k] = instantiate(cfg.runner.tune.training[k])
                if trainhp[k].get('grid_search', None) is not None:
                    trainhp[k]['grid_search'] = list(trainhp[k]['grid_search'])
            print('Configured training:', trainhp)

    tune_args['config'] = {"training": {**cfg.training, **trainhp}, "model": {**cfg.model, **modelhp}, **cfg.runner.trainable, **cfg.dataset}
    tune_args['config']['data_loader'] = data_loader
    print('starting tune with config', tune_args['config'])
    analysis = tune.run(TuneTrainer,
        **tune_args, trial_dirname_creator=lambda p: 'TuneTrainer_'+p.trial_id
    )
    print('found best params: ', analysis.best_config)
    return analysis
