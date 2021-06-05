import tensorflow as tf
from tensorflow.keras.layers import InputLayer, BatchNormalization, Conv2D, Conv2DTranspose, Lambda,AveragePooling2D, MaxPooling2D, Dense
from tensorflow.python.keras.layers.core import Reshape
from data.default_loader import get_preprocessing_layers
import copy
from .param_utils import get_array_from_params_split
from hydra.experimental import compose, initialize, initialize_config_dir


class ChannelWiseDense(tf.keras.layers.Layer):
    def __init__(self, input_shape, activation, **kwargs):
        super().__init__(trainable=True, name="channelwise", **kwargs)
        self.input_shape_ = input_shape
        self.layers = []
        for i in range(input_shape[-1]):
            self.layers.append(Dense(input_shape[0] * input_shape[1], activation=activation))


    def build(self, input_shape):
        for l in self.layers:
            l.build((input_shape[0], input_shape[1] * input_shape[2]))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        out = None
        # reshape to (batch, x * y, f)
        x = tf.reshape(inputs, tf.constant([-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]], dtype=tf.int32))
        for i in range(len(self.layers)):
            # reshape to (batch, x, y, 1)
            d_out = tf.reshape(self.layers[i](x[:,:,i]), tf.constant([-1, inputs.shape[1], inputs.shape[2]], dtype=tf.int32))
            d_out = tf.expand_dims(d_out, -1)
            if out is not None:
                out = tf.concat([d_out, out],axis=-1)
            else:
                out = d_out
        return out

class Autoenc_light2(tf.keras.Model):
    def __init__(self, ds_mean = None, params: dict = {}):
        super().__init__()
        if ds_mean is None:
            ds_mean = [0,0,0]
        
        params = copy.deepcopy(params)

        params['e_activation'] = get_array_from_params_split(params['e_activation'])
        params['e_filters'] = get_array_from_params_split(params['e_filters'])
        params['e_h_activation'] = get_array_from_params_split(params['e_h_activation'])
        params['d_filters'] = get_array_from_params_split(params['d_filters'])
        params['d_activation'] = get_array_from_params_split(params['d_activation'])

        self.preprocess = get_preprocessing_layers(ds_mean)

        encoder_layers = [
                    InputLayer(input_shape=(96, 96, 3)),
                    Conv2D(filters=params['e_filters'][0], kernel_size=3, strides=(1, 1), activation=params['e_activation'], padding='same',
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                    Conv2D(filters=params['e_filters'][1], kernel_size=3, strides=(1, 1), activation=params['e_activation'], padding='same',
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                    Conv2D(filters=params['e_filters'][2], kernel_size=3, strides=(1, 1), activation=params['e_activation'], padding='same',
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                    Conv2D(filters=params['e_filters'][3], kernel_size=3, strides=(1, 1), activation=params['e_activation'], padding='same',
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                    Conv2D(filters=params['e_filters'][4], kernel_size=3, strides=(1, 1), activation=params['e_activation'], padding='same',
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                    
                    Conv2D(filters=params['e_filters'][5], kernel_size=3, strides=(2, 2), activation=params['e_activation'],
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                    Conv2D(filters=params['e_filters'][6], kernel_size=3, strides=(2, 2), activation=params['e_activation'],
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),

                    Conv2D(filters=params['e_filters'][7], kernel_size=5, strides=(2, 2), activation=params['e_activation'],
                    kernel_initializer=params['e_initializer'], bias_initializer=params['e_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),

                    Reshape((10*10*params['e_filters'][7],)),
                    Dense(10*10*params['e_filters'][7], activation='linear'),
                    Dense(6*6*params['e_filters'][7], activation='linear'),
                    Reshape((6,6,params['e_filters'][7],)),

                    Conv2D(filters=params['e_filters'][8], kernel_size=3, strides=(1, 1), activation=params['e_h_activation'], padding='same',
                    kernel_initializer=params['e_h_initializer'], bias_initializer=params['e_h_initializer'],
                    kernel_regularizer=self._try_get_regularizer_encoder(params), bias_regularizer=self._try_get_regularizer_encoder(params)),
                ]
        if params['batch_norm'] == True:
            i = 2
            while i < 18:
                encoder_layers.insert(i, BatchNormalization())
                i += 2
            encoder_layers.insert(20, BatchNormalization())
            

        self.encoder = tf.keras.Sequential(encoder_layers)
        #self.encoder_preview = tf.keras.Model(inputs=self.encoder.input, outputs=self.encoder.get_layer("latent").output)

        #(1,1,channels) tensor with mean of each channel -
        # should contain values obtained after featurewise center preprocessing
        self.center_mean = tf.Variable([[ds_mean]], dtype=tf.float32)

        dec_shape = (self.encoder.output.shape[1], self.encoder.output.shape[2], self.encoder.output.shape[3])
        self.decoder1 = tf.keras.Sequential(
            [
                InputLayer(input_shape=(dec_shape[0],dec_shape[1],dec_shape[2])),
                Conv2DTranspose(params['d_filters'][0], kernel_size=3, strides=(1,1), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),

                Conv2DTranspose(params['d_filters'][1], kernel_size=3, strides=(2,2), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),
                Conv2DTranspose(params['d_filters'][2], kernel_size=3, strides=(2,2), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),
                Conv2DTranspose(params['d_filters'][3], kernel_size=3, strides=(2,2), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),

                Conv2DTranspose(3, kernel_size=5, strides=(2,2), activation='sigmoid', padding='same',
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),
            ]
        )
        self.decoder2 = tf.keras.Sequential(
            [
                InputLayer(input_shape=(dec_shape[0],dec_shape[1],dec_shape[2])),
                Conv2DTranspose(params['d_filters'][0], kernel_size=3, strides=(1,1), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),

                Conv2DTranspose(params['d_filters'][1], kernel_size=3, strides=(2,2), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),
                Conv2DTranspose(params['d_filters'][2], kernel_size=3, strides=(2,2), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),
                Conv2DTranspose(params['d_filters'][3], kernel_size=3, strides=(2,2), activation=params['d_activation'], padding='same',
                    kernel_initializer=params['d_initializer'], bias_initializer=params['d_initializer'],
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),

                Conv2DTranspose(3, kernel_size=5, strides=(2,2), activation='sigmoid', padding='same',
                    kernel_regularizer=self._try_get_regularizer_decoder(params), bias_regularizer=self._try_get_regularizer_decoder(params)),
            ]
        )

    def _try_get_regularizer_encoder(self, params: dict):
        if params.get('e_regularizer') == True:
            return tf.keras.regularizers.L2(0.0005)

    def _try_get_regularizer_decoder(self, params: dict):
        if params.get('d_regularizer') == True:
            return tf.keras.regularizers.L2(0.0005)

    def set_center_mean(self, mean_values):
        self.center_mean.assign_add(mean_values)

    def get_original(self, encoder_output):
        return encoder_output + self.center_mean

    def call(self, inputs, training=False, p1=False, p2=False):
        i = self.preprocess(inputs, training=training)
        h = self.encoder(i,training=training)
        if not p1 and not p2:
            y1 = self.decoder1(h,training=training)
            y2 = self.decoder2(h,training=training)
        if p1:
            return self.decoder1(h,training=training)
        elif p2:
            return self.decoder2(h,training=training)
        return (y1,y2)



if __name__ == '__main__':
    with initialize(config_path='../conf'):
        cfg = compose(config_name="config", overrides=['model=autoenc_light2'])
    model = Autoenc_light2(params=cfg['model'])
    model.build((1,96,96,3))
    model.encoder.summary()
    model.decoder1.summary()
    model.decoder2.summary()