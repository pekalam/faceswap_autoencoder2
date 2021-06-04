import os

import tensorflow as tf


class TensorboardLogger():
    def __init__(self, logdir: str):
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = tf.summary.create_file_writer(logdir)

    def begin_train_step():
        pass

    def log_metrics(self, result_dict: dict, step):
        tf.summary.experimental.set_step(step)
        with self.writer.as_default():
            for k,v in result_dict.items():
                tf.summary.scalar(k, v) #log metrics
            #for x,y in self.p1_iter:
            #    h1 = self.model.encoder(x)
            #    y2 = self.model.decoder2(h1)
            #tf.summary.scalar("Input image loss") #show preview
