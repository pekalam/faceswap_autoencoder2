import sys
import os
from main import _load_cfg_from_argv, main
from tests.utils import read_default_config


def test_load_from_absolute_path():
    config_path = os.path.join(os.getcwd(), "tests", "config_test")
    sys.argv = ["--absoulute-config-path="+config_path]
    (absolute_config, model) = _load_cfg_from_argv()
    assert absolute_config is not None
    assert absolute_config.test1 == 1
    assert absolute_config.test2 == 2

def test_makes_train_pass_without_error():
    cfg = read_default_config()
    cfg.training.max_iterations = 1
    main(cfg)