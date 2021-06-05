from data.default_loader import FaceDsLoader
import numpy as np
from .utils import read_default_config
import pytest


@pytest.mark.parametrize("val_split,val_batch_size,test_batch_size,batch_size", [(0, 0, 1, 1), (0.2, 1, 1, 10), (0.2, None, None, 1)])
def test_loads_valid_images(val_split, val_batch_size, test_batch_size, batch_size):

    for ds_path_p1, ds_path_p2, has_validation in [("__dataset/serena/", "__dataset/novak/", False), ("__dataset2_13/serena/", "__dataset2_13/novak/", True), ("__dataset2_13-rev1/serena/", "__dataset2_13-rev1/novak/", True), 
    ("__dataset3/serena/", "__dataset3/novak/", True), ("__dataset3_masked/serena/", "__dataset3_masked/novak/", True), ("__dataset3_masked_large/serena/", "__dataset3_masked_large/novak/", True)]:
        if has_validation and val_split > 0:
            continue
        loader = FaceDsLoader(ds_path_p1, ds_path_p2, None if not has_validation else {
                              'has_validation': True})
        config = read_default_config()
        config.training.batch_size = batch_size
        config.training.val_split = val_split
        config.training.val_batch_size = val_batch_size
        config.training.test_batch_size = test_batch_size
        ds = loader(config)
        assert ds.x_p1 is not None and ds.x_p2 is not None
        p1_el = next(iter(ds.x_p1))
        p2_el = next(iter(ds.x_p2))
        assert p1_el[0].shape == p2_el[0].shape
        assert type(p1_el) is tuple and type(p2_el) is tuple
        assert p1_el[0].shape[0] == config.training['batch_size'] and p2_el[0].shape[0] == config.training['batch_size']
        assert len(ds.x_p1) > 0 and len(
            ds.x_p2) > 0 and len(ds.x_p1) == len(ds.x_p2)

        if ds.x_p1val is not None and ds.x_p2val is not None:
            p1val_el = next(iter(ds.x_p1val))
            p2val_el = next(iter(ds.x_p2val))
            assert len(ds.x_p1val) > 0 and len(
                ds.x_p2val) > 0 and len(ds.x_p1val) == len(ds.x_p2val)
            assert type(p1val_el) is tuple and type(p2val_el) is tuple
            if val_batch_size == None or val_batch_size == 0:
                assert p1val_el[0].shape[0] > 0 and p2val_el[0].shape[0] > 0
            else:
                assert p1val_el[0].shape[0] == val_batch_size and p2val_el[0].shape[0] == val_batch_size
        else:
            assert ds.x_p1val is None and ds.x_p2val is None and has_validation == False

        if ds.x_p1test is not None and ds.x_p2test is not None:
            p1test_el = next(iter(ds.x_p1test))
            p2test_el = next(iter(ds.x_p2test))
            assert len(ds.x_p1test) > 0 and len(
                ds.x_p2test) > 0 and len(ds.x_p1test) == len(ds.x_p2test)
            assert type(p1test_el) is tuple and type(p2test_el) is tuple
            if test_batch_size == None:
                assert p1test_el[0].shape[0] > 0 and p2test_el[0].shape[0] > 0
            else:
                assert p1test_el[0].shape[0] == test_batch_size and p2test_el[0].shape[0] == test_batch_size
        else:
            assert ds.x_p1test is None and ds.x_p2test is None
        assert len(ds.x_p1prev) > 0 and len(ds.x_p2prev) > 0
        assert len(ds.x_p1prev) == 3 and len(ds.x_p2prev) == 3
        assert len(ds.x_p1prev[0].shape) == 4 and len(
            ds.x_p2prev[0].shape) == 4
        assert ds.x_p1prev[0].shape[0] == 1 and ds.x_p2prev[0].shape[0] == 1

        def val_generator(gen):
            for x, y in gen:

                assert x.shape == y.shape
                assert not np.array_equal(x, y)
                break
        val_generator(ds.x_p1)
        val_generator(ds.x_p2)
