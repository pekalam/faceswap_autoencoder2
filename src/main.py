import hydra
from omegaconf import DictConfig, OmegaConf
from utils.config_utils import get_compatible_config
from hydra.experimental import compose, initialize, initialize_config_dir
import sys
from main_impl import main_impl 
import json

org_cfg = None
secrets = None


def _load_org_config(model=None):
    global org_cfg
    with initialize(config_path='conf'):
        org_cfg = compose(config_name="config", overrides=['model='+model] if model is not None else [])
        OmegaConf.set_struct(org_cfg, True)


def _load_cfg_from_argv():
    custom_path = [a.split('=')[1] for a in sys.argv if '--config-path' in a]
    model = None
    if len(custom_path) > 0:
        with initialize(config_path=custom_path[0]):
            cfg = compose(config_name="config")
            OmegaConf.set_struct(cfg, True)
            model = cfg.model.name
    absoulute_config_arg = [(a,a.split('=')[1]) for a in sys.argv if '--absoulute-config-path' in a]
    absolute_config = None
    if len(absoulute_config_arg) > 0:
        absoulute_config_arg = absoulute_config_arg[0]
        sys.argv.pop(sys.argv.index(absoulute_config_arg[0]))
        with initialize_config_dir(config_dir=absoulute_config_arg[1]):
            absolute_config = compose(config_name="config")
            model = absolute_config.model.name
            OmegaConf.set_struct(absolute_config, True)
    return (absolute_config, model)

def _load_secrets_from_argv():
    global secrets
    secrets_arg = [(a,a.split('=')[1]) for a in sys.argv if '--secrets-path' in a]
    if len(secrets_arg) == 0:
        return
    secrets_arg = secrets_arg[0]
    sys.argv.pop(sys.argv.index(secrets_arg[0]))
    with open(secrets_arg[1]) as f:    
        secrets = json.load(f)
    print('loaded secrets from path ', secrets_arg[1])

@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    global org_cfg, secrets
    cfg = get_compatible_config(cfg, org=org_cfg)
    print(OmegaConf.to_yaml(cfg))
    main_impl(cfg, secrets)


def notebook_main(cfg: DictConfig):
    _load_org_config(cfg.model.name)
    _load_secrets_from_argv()
    main(cfg)

if __name__ == '__main__':
    (absolute_config, model) = _load_cfg_from_argv()
    _load_secrets_from_argv()
    _load_org_config(model)
    if absolute_config is not None:
        main(absolute_config)
    else:
        main()
