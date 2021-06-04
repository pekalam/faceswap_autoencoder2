
from model.impl_factory import model_impl_factory as get_autoenc_impl_from_config
from training.preview_creator import PreviewCreator
from .early_stopping import EarlyStopping
import tensorflow as tf
import os
import collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from training.trainer_base import TrainerBase
import time

def tee_q_sz(iterable, n=2, q_sz=None):
    it = iter(iterable)
    deques = [collections.deque(maxlen=q_sz) for _ in range(n)]
    def gen(mydeque):
        while True:
            if not mydeque:             # when the local deque is empty
                try:
                    newval = next(it)   # fetch a new value and
                except StopIteration:
                    return
                for d in deques:        # load it to all the deques
                    if d != mydeque and len(d) < q_sz:
                        d.append(newval)
            else: yield mydeque.popleft()
            yield newval
    return tuple(gen(d) for d in deques)

class FlipLayer(tf.keras.layers.Layer):
    def __init__(self, seed, **kwargs):
        super(FlipLayer, self).__init__(**kwargs)
        self.rnd = tf.random.Generator.from_seed(seed)
    def call(self, inputs, **kwargs):
        flip = self.rnd.uniform([1], 0., 1.)
        return tf.cond(tf.less(flip, 0.5), lambda: tf.image.flip_left_right(inputs), lambda: inputs)

class FaceSwapTrainer(TrainerBase):
    def __init__(self, cfg: dict, data_loader = None, logger = None, saver = None, load_latest_checkpoint = False, preview_logdir: str = None, model=None, data_loader_output=None):
        super(FaceSwapTrainer, self).__init__(cfg, data_loader, model, data_loader_output)
        self.cfg = cfg
        self.logger = logger
        self.saver = saver

        #step variable for tf checkpoint, logger compatibility
        self.step = tf.Variable(0, dtype=tf.int64)

        #load person1, person2 generators
        self.data_loader = data_loader
        self.ds = data_loader(cfg) if data_loader is not None else data_loader_output
        self.model = get_autoenc_impl_from_config(cfg, ds_mean=data_loader.ds_mean) if model is None else model
        self._build_model_and_show_summary()
        #assuming lenghts are equal
        self.total_iter=len(self.ds.x_p1)

        if self.saver is not None:
            self.saver.initialize_saver(model=self.model,step=self.step)
            if load_latest_checkpoint:
                self.saver.restore_latest(self.mgr.latest_checkpoint)

        lr = self.cfg['training']['learning_rate']
        early_stop_patience = cfg['training']['early_stop_patience']
        early_stop_min_delta_patience = cfg['training']['early_stop_min_delta_patience']
        early_stop_min_delta = cfg['training']['early_stop_min_delta']

        self.checkpoint_iter = 0
        self.early_stopping = EarlyStopping(early_stop_patience, early_stop_min_delta, early_stop_min_delta_patience)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr) if self.cfg['training'].get('optimizer', None) is None else instantiate(self.cfg['training']['optimizer'], learning_rate=lr)
        self.train_loss = tf.keras.losses.MeanAbsoluteError()
        self.preview_creator = PreviewCreator(self.cfg['training']['img_preview_threshold'], self.ds.x_p1prev, self.ds.x_p2prev, preview_logdir)
        if cfg['training']['random_flip']:
            self.flip_layer_x = tf.keras.models.Sequential([FlipLayer(seed=self.seed,trainable=False)])
            self.flip_layer_y = tf.keras.models.Sequential([FlipLayer(seed=self.seed,trainable=False)])
        else:
            self.flip_layer_x = tf.keras.models.Sequential([])
            self.flip_layer_y = tf.keras.models.Sequential([])


    def _build_model_and_show_summary(self):
        input_shape = eval(self.cfg['model']['input_shape'])
        print("Building model wit input_shape ", input_shape)
        #crucial step for loading checkpoint
        self.model.build(input_shape)
        self.model.summary()
        if hasattr(self.model, 'encoder'):
            self.model.encoder.summary()
        if hasattr(self.model, 'decoder1'):
            self.model.decoder1.summary()
        if hasattr(self.model, 'decoder2'):
            self.model.decoder2.summary()

    def try_save_checkpoint(self):
        if self.saver is not None:
            self.saver.save(self.step)

    #@tf.function
    def _calc_loss(self, labels, predictions, batch_size):
        per_example_loss = self.train_loss(labels, predictions)
        if tf.rank(per_example_loss) < 1:
            per_example_loss = tf.expand_dims(per_example_loss, axis=0)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    #@tf.function
    def _train_step_p1(self, x, y):
        batch_loss = 0.
        with tf.GradientTape() as tape:
            y_net = self.model(x, training=True, p1=True, p2=False)
            if tf.distribute.has_strategy():
                batch_loss = self._calc_loss(y, y_net, self.cfg['training']['global_batch_size'])
            else:
                batch_loss = self.train_loss(y, y_net)
        grads = tape.gradient(batch_loss, self.model.trainable_variables, unconnected_gradients='zero')
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return batch_loss, grads
    
    #@tf.function
    def _train_step_p2(self, x, y):
        batch_loss = 0.
        with tf.GradientTape() as tape:
            y_net = self.model(x, training=True, p1=False, p2=True)
            if tf.distribute.has_strategy():
                batch_loss = self._calc_loss(y, y_net, self.cfg['training']['global_batch_size'])
            else:
                batch_loss = self.train_loss(y, y_net)
        grads = tape.gradient(batch_loss, self.model.trainable_variables, unconnected_gradients='zero')
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return batch_loss, grads

    #@tf.function
    def _train_epoch(self):
        total_loss_train = 0.0
        num_batches = 0
        loss_train = 0.

        iterations = 0

        while iterations < self.total_iter:
            x,y = next(self.p1_iter)
            x = self.flip_layer_x(x)
            y = self.flip_layer_y(y)
            loss_train, grads_p1 = self._train_step_p1(x, y)
            total_loss_train += loss_train
            num_batches = num_batches + 1

            x,y = next(self.p2_iter)
            x = self.flip_layer_x(x)
            y = self.flip_layer_y(y)
            loss_train, grads_p2 = self._train_step_p2(x, y)
            total_loss_train += loss_train
            num_batches = num_batches + 1

            iterations += 1
        
        total_loss_train = total_loss_train / tf.cast(num_batches, tf.float32)
        return total_loss_train, grads_p1, grads_p2


    def _val_step(self):
        total_loss = 0.
        num_batches = 0
        p1_loss = 0.
        #for x,y in tqdm(self.val_ds):
        for x,y in self.ds.x_p1val:
            y_net = self.model(x, training=False, p1=True, p2=False)
            batch_loss = self.train_loss(y, y_net)
            p1_loss += batch_loss
            num_batches = num_batches + 1
        p1_loss = p1_loss / tf.cast(num_batches, tf.float32)
        total_loss += p1_loss

        num_batches = 0
        p2_loss = 0.
        #for x,y in tqdm(self.val_ds):
        for x,y in self.ds.x_p2val:
            y_net = self.model(x, training=False, p1=False, p2=True)
            batch_loss = self.train_loss(y, y_net)
            p2_loss += batch_loss
            num_batches = num_batches + 1
        p2_loss = p2_loss / tf.cast(num_batches, tf.float32)
        total_loss += p2_loss

        total_loss /= 2.

        return p1_loss, p2_loss, total_loss

    def train_step(self):
        if self.logger is not None:
            self.logger.begin_train_step()

        start = time.time()

        self.p1_iter = iter(self.ds.x_p1)
        self.p2_iter = iter(self.ds.x_p2)

        print('starting train step')
        train_loss, grads_p1, grads_p2 = self._train_epoch()
        print('step finished')

        early_stop_metric = train_loss

        if self.cfg['training']['show_gradients']:
            p1 = {}
            for i in range(len(grads_p1)):
                if 'kernel' in self.model.trainable_variables[i].name:
                    p1['p1_grad'+str(i)] = [grads_p1[i]]
            p2 = {}
            for i in range(len(grads_p2)):
                if 'kernel' in self.model.trainable_variables[i].name:
                    p2['p2_grad'+str(i)] = [grads_p2[i]]
            
        result = {
            "loss": train_loss.numpy(),
            "grads_p1": None if not self.cfg['training']['show_gradients'] else p1,
            "grads_p2": None if not self.cfg['training']['show_gradients'] else p2,
        }
        print(result['loss'])


        del self.p1_iter
        del self.p2_iter

        if self.ds.x_p1val is not None and self.ds.x_p2val is not None:
            print('starting val step')
            p1_val_loss, p2_val_loss, total_val_loss = self._val_step()
            print('val step finished')
            result = {
                **result,
                "p1_val_loss": p1_val_loss.numpy(),
                "p2_val_loss": p2_val_loss.numpy(),
                "val_loss": total_val_loss.numpy()
            }
            print(result['val_loss'])
            early_stop_metric = total_val_loss

        if self.step == 0:
            result['mean'] = self.data_loader.ds_mean
            result['seed'] = self.seed

        self.step.assign_add(1)
        print('epoch ', self.step.numpy(), ' finished')
        print('epochs per h: ', (60*60)/(time.time()-start))
        
        if self.logger is not None:
            self.logger.log_metrics(result, self.step)

        #val loss early stopping
        should_checkpoint, should_stop = self.early_stopping.add_metric(early_stop_metric)
        result['stop'] = should_stop
        if should_stop:
            return result
        if should_checkpoint:
            self.try_save_checkpoint()
            result['checkpoint'] = True

        
        #manual checkpoint
        self.checkpoint_iter += 1
        if not self.early_stopping.metric_increased:
            if self.checkpoint_iter >= self.cfg['training']['checkpoint_frequency']:
                print('saving checkpoint')
                self.try_save_checkpoint()
                result['checkpoint'] = True
                self.checkpoint_iter = 0
        elif self.checkpoint_iter >= self.cfg['training']['checkpoint_frequency']:
            print('checkpoint not saved due to val loss increase')
            

        #preview
        img, img_p1, img_p2 = self.preview_creator.append_step(self.model, self.step)
        if img is not None:
            Image.fromarray(np.uint8(img*255)).save('example.jpg')
            result['image'] = img
            result['image_p1'] = img_p1
            result['image_p2'] = img_p2

                


        if result['loss'] <= self.cfg['training']['target_loss']:
            print('Reached target loss')
            self.try_save_checkpoint()
            result['stop'] = True
            result['checkpoint'] = True
            return result

        #max iter condition
        if self.cfg['training']['max_iterations'] is not None and self.step >= self.cfg['training']['max_iterations']:
            print('Reached max iters')
            self.try_save_checkpoint()
            result['stop'] = True
            result['checkpoint'] = True
            return result

        return result


