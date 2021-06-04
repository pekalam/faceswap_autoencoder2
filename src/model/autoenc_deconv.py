import tensorflow as tf
from tensorflow.keras.layers import InputLayer, BatchNormalization, Conv2D, Conv2DTranspose, Lambda,AveragePooling2D, MaxPooling2D
from data.default_loader import get_preprocessing_layers

class Autoenc_deconv(tf.keras.Model):
    def __init__(self, ds_mean = None, params: dict = {}):
        super().__init__()
        if ds_mean is None:
            ds_mean = [0,0,0]
        
        self.preprocess = get_preprocessing_layers(ds_mean)

        self.encoder = tf.keras.Sequential([
                    InputLayer(input_shape=(96, 96, 3)),
                    Conv2D(
                        filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='same'),
                    MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'),
                    #Conv2D(
                    #    filters=conv1_f, kernel_size=3, strides=(1, 1), activation=conv1_a, padding=conv1_p),
                    Conv2D(
                        filters=64, kernel_size=2, strides=(1, 1), activation='relu', padding='same'),
                    BatchNormalization(),
                    Conv2D(
                        filters=128, kernel_size=2, strides=(1, 1), activation='relu', padding='same'),
                    BatchNormalization(),
                    Conv2D(
                        filters=128, kernel_size=2, strides=(2, 2), activation='relu', padding='same'),
                    AveragePooling2D(pool_size=(2, 2), strides=(2,2)),
                    BatchNormalization(),
                    Conv2D(
                        filters=64, kernel_size=2, strides=(1, 1), activation='tanh', padding='same', name="latent"),
                ])
        #self.encoder_preview = tf.keras.Model(inputs=self.encoder.input, outputs=self.encoder.get_layer("latent").output)

        #(,channels) tensor with mean of each channel -
        # should contain values obtained after featurewise center preprocessing
        self.center_mean = tf.Variable(ds_mean, dtype=tf.float32)

        dec_shape = (self.encoder.output.shape[1], self.encoder.output.shape[2], self.encoder.output.shape[3])
        self.decoder1 = tf.keras.Sequential(
            [
                InputLayer(input_shape=(dec_shape[0],dec_shape[1],dec_shape[2])),
                BatchNormalization(),
                Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), activation='tanh'),
                BatchNormalization(),
                Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), activation='tanh'),
                BatchNormalization(),
                Conv2DTranspose(3, kernel_size=(2,2), strides=(2,2), activation='sigmoid')
            ]
        )
        self.decoder2 = tf.keras.Sequential(
            [
                InputLayer(input_shape=(dec_shape[0],dec_shape[1],dec_shape[2])),
                BatchNormalization(),
                Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), activation='tanh'),
                BatchNormalization(),
                Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), activation='tanh'),
                BatchNormalization(),
                Conv2DTranspose(3, kernel_size=(2,2), strides=(2,2), activation='sigmoid')
            ]
        )

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
    model = Autoenc_deconv()
    model.build((1,96,96,3))
    model.encoder.summary()
    model.decoder1.summary()
    model.decoder2.summary()