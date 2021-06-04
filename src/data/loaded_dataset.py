import tensorflow as tf

class LoadedDataset():
    """Contains dataset loaded by calling DatasetLoader object __call__ method
    """
    def __init__(self, x_p1, x_p2, x_p1val = None, x_p2val = None, x_p1test = None, x_p2test = None, x_p1prev = None, x_p2prev = None):
        self._x_p1 = x_p1
        self._x_p2 = x_p2
        self._x_p1_val = x_p1val
        self._x_p2_val = x_p2val
        self._x_p1_test = x_p1test
        self._x_p2_test = x_p2test
        self._x_p1_prev = x_p1prev
        self._x_p2_prev = x_p2prev

    @property
    def x_p1(self) -> tf.data.Dataset: return self._x_p1

    @property
    def x_p2(self) -> tf.data.Dataset: return self._x_p2

    @property
    def x_p1val(self) -> tf.data.Dataset: return self._x_p1_val

    @property
    def x_p2val(self) -> tf.data.Dataset: return self._x_p2_val

    @property
    def x_p1test(self) -> tf.data.Dataset: return self._x_p1_test
    
    @property
    def x_p2test(self) -> tf.data.Dataset: return self._x_p2_test

    @property
    def x_p1prev(self): return self._x_p1_prev
    
    @property
    def x_p2prev(self): return self._x_p2_prev