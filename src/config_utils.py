import json
import os
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose
from hydra.core.global_hydra import GlobalHydra

def save_as_one_file(cfg: DictConfig, file_path: str):
    content = OmegaConf.to_yaml(cfg)
    with open(file_path, 'w') as f:
        f.writelines(content)


def get_compatible_config(cfg: DictConfig, org: DictConfig = None):
    def count(d):
        return sum([count(v)+1 if isinstance(v, dict) or isinstance(v,DictConfig) else 1 for v in d.values()])
    if org is None:
        org: dict = GlobalHydra.instance().config_loader().load_configuration(config_name='config', overrides=['model='+cfg.model.name], run_mode=RunMode.RUN)
    org = {**org}
    if org.get('hydra', None) is not None:
        org.pop('hydra')
    compat = {**org, **cfg,"runner": {**org['runner'], **cfg['runner']}, "model": {**org['model'], **cfg['model']},
            "training": {**org['training'], **cfg['training']},"dataset": {**org['dataset'], **cfg['dataset']}}
    if count(cfg) != count(org):
        print("Provided config were made compatible due to different items count")
    #debug
    #print(count(cfg), count(org))
    #print(org)
    #print(cfg)
    compat_dc = DictConfig(compat)
    return compat_dc

def read_experiment_params(artifacts_dir_path):
    params_path = os.path.join(artifacts_dir_path, "params.json")
    if not os.path.exists(params_path):
        raise ValueError("cannot find params.json at" + params_path)

    with open(params_path) as f:
        params = json.load(f)
    print('read params: ', params)
    return params