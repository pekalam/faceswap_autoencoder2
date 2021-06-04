import numpy as np
import tensorflow as tf


class PreviewCreator:
    def __init__(self, threshold, prev_p1, prev_p2, logdir: str = None):
        self.preview_iter = 0
        self.threshold = threshold
        self.prev_p1 = prev_p1
        self.prev_p2 = prev_p2
        if logdir is not None:
            self.writer = tf.summary.create_file_writer(logdir)
        else:
            self.writer = None

    def append_step(self, model, step) -> np.ndarray:
        self.preview_iter += 1
        if not self.preview_iter == self.threshold:
            return None, None, None
        else:
            print('creating img preview')
            img = np.ones(( (20+self.prev_p1[0].shape[1])*len(self.prev_p1)*2+20, 20+self.prev_p1[0].shape[2]+20+self.prev_p2[0].shape[2]+20, self.prev_p1[0].shape[3] ))
            i=0
            for x in self.prev_p1:
                y2 = model(x, p2=True)
                img[(20+self.prev_p1[0].shape[1])*i+20 : (20+self.prev_p1[0].shape[1])*i+20+self.prev_p1[0].shape[1], 20 : 20+self.prev_p1[0].shape[2], :]=y2[0]
                i += 1
            for x in self.prev_p1:
                img[(20+self.prev_p1[0].shape[1])*i+20 : (20+self.prev_p1[0].shape[1])*i+20+self.prev_p1[0].shape[1], 20 : 20+self.prev_p1[0].shape[2], :]=x[0]/255.0
                i += 1
            i=0
            for x in self.prev_p2:
                y1 = model(x, p1=True)
                img[(20+self.prev_p1[0].shape[1])*i+20 : (20+self.prev_p1[0].shape[1])*i+20+self.prev_p1[0].shape[1], 20+self.prev_p1[0].shape[2]+20 : 20+self.prev_p1[0].shape[2]+20+self.prev_p2[0].shape[2], :]=y1[0]
                i += 1
            for x in self.prev_p2:
                img[(20+self.prev_p1[0].shape[1])*i+20 : (20+self.prev_p1[0].shape[1])*i+20+self.prev_p1[0].shape[1], 20+self.prev_p1[0].shape[2]+20 : 20+self.prev_p1[0].shape[2]+20+self.prev_p2[0].shape[2], :]=x[0]/255.0
                i += 1
            self.preview_iter = 0
            img_p1 = model(self.prev_p1[0], p1=True)
            img_p2 = model(self.prev_p2[0], p2=True)
            
            img_test_p1 = model(self.prev_p1[-1], p1=True)
            img_test_p2 = model(self.prev_p2[-1], p2=True)
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.image('Preview img', np.expand_dims(img, axis=0), step)
                    tf.summary.image('train p1 output', img_p1, step)
                    tf.summary.image('train p2 output', img_p2, step)

                    tf.summary.image('val p1 output', img_test_p1, step)
                    tf.summary.image('val p2 output', img_test_p2, step)
            
            return img, img_p1, img_p2