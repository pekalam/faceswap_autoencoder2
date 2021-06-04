import tensorflow as tf


class TensorflowSaver:
    def __init__(self, logdir: str):
        self.logdir = logdir

    def initialize_saver(self,**kwargs):
        self.ckpt = tf.train.Checkpoint(**kwargs)
        self.mgr = tf.train.CheckpointManager(
            self.ckpt, self.logdir, max_to_keep=5)
    
    def save(self, step):
        self.mgr.save(step)

    def restore_latest(self):
        self.mgr.restore(self.mgr.latest_checkpoint)