import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision, AUC, BinaryAccuracy
import tensorflow.keras.backend as K


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def model1d(n_kernels=128):
    l1 = 1e-9
    l2 = 1e-9

    loss = tf.keras.losses.binary_crossentropy

    model = tf.keras.Sequential(name='model1d')
    model.add(tf.keras.layers.BatchNormalization(input_shape=(3000, 16)))

    model.add(tf.keras.layers.Conv1D(filters=n_kernels // 4,
                                     kernel_size=5,
                                     dilation_rate=7,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.MaxPool1D(5))
    model.add(tf.keras.layers.Dropout(.2))

    model.add(tf.keras.layers.Conv1D(filters=n_kernels // 2,
                                     kernel_size=5,
                                     dilation_rate=5,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.MaxPool1D(3))
    model.add(tf.keras.layers.Dropout(.2))

    for i in range(2):
        model.add(tf.keras.layers.Conv1D(filters=(n_kernels // 4) * (3 + i),
                                         kernel_size=3,
                                         dilation_rate=3,
                                         padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.MaxPool1D(2))
        model.add(tf.keras.layers.Dropout(0.2))

    n_k = {0: n_kernels,
           1: (n_kernels // 4) * 3,
           2: n_kernels // 2,
           3: n_kernels // 4,
           4: n_kernels // 4,
           5: n_kernels // 4,
           }

    for i in range(3):
        model.add(tf.keras.layers.Conv1D(filters=n_k[2*i],
                                         kernel_size=4,
                                         dilation_rate=1,
                                         padding='valid',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
        model.add(tf.keras.layers.Conv1D(filters=n_k[2*i+1],
                                         kernel_size=4,
                                         dilation_rate=1,
                                         padding='valid',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.MaxPool1D(2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(.5))
    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[AUC(curve='ROC', name='roc_auc'), AUC(curve='PR', name='pr_auc'), Precision(), Recall(), mean_pred, BinaryAccuracy()])

    print(model.summary())
    return model


# adapted from tf2_matthias/models/iic_resnet_backbone.py
def resnet1d(n_kernels=32):
    kernel_size = 5

    def residual_block(X, kernels, size):
        out = tf.keras.layers.Conv1D(kernels, size, padding='same')(X)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv1D(kernels, size, padding='same')(out)
        out = tf.keras.layers.add([X, out])
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.MaxPool1D(pool_size=5, strides=3)(out)
        return out

    # see here: https://www.tensorflow.org/guide/distributed_training
    # NOTE: Use tf2 virtual environment locally.
    # For debugging one can see a verbose run with export TF_CPP_MIN_LOG_LEVEL='0' && python3 run.py
    strategy = tf.distribute.MirroredStrategy(devices=None)  # on all available GPUs  # works! # choose which GPUs shall be chosen with i.e.  export CUDA_VISIBLE_DEVICES=0,1,2,3,4 - uses then all five available GPUs
    # strategy = tf.distribute.get_strategy()  # default strategy == no strategy  # works

    with strategy.scope():
        loss = tf.keras.losses.binary_crossentropy

        inputs = tf.keras.layers.Input([3000, 16])
        x = tf.keras.layers.Conv1D(filters=n_kernels, kernel_size=kernel_size)(inputs)
        x = residual_block(x, n_kernels, kernel_size)
        x = residual_block(x, n_kernels, kernel_size)
        x = residual_block(x, n_kernels, kernel_size)
        x = residual_block(x, n_kernels, kernel_size)
        x = residual_block(x, n_kernels, kernel_size)
        x = residual_block(x, n_kernels, kernel_size)
        output = tf.keras.layers.GlobalMaxPool1D()(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        model = tf.keras.Model(inputs=inputs, outputs=output, name="resnet1d")

        model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[AUC(curve='ROC', name='roc_auc'), AUC(curve='PR', name='pr_auc'), Precision(), Recall(), mean_pred, BinaryAccuracy()])

    # print(model.summary())
    return model


def resnet1d_v20201002(n_kernels=16):
    kernel_size = 5
    l1 = 1e-9
    l2 = 1e-9

    def residual_block(X, kernels_conv1, kernels_conv2, kernel_size, dilation):
        out = tf.keras.layers.Conv1D(filters=kernels_conv1,
                                     kernel_size=kernel_size,
                                     dilation_rate=dilation,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)
                                     )(X)
        out = tf.keras.layers.add([X, out])
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.LeakyReLU(0.2)(out)
        out = tf.keras.layers.Conv1D(filters=kernels_conv2,
                                     kernel_size=kernel_size,
                                     dilation_rate=dilation,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)
                                     )(X)
        out = tf.keras.layers.MaxPool1D(2)(out)
        out = tf.keras.layers.Dropout(.2)(out)
        return out

    # see here: https://www.tensorflow.org/guide/distributed_training
    # NOTE: Use tf2 virtual environment locally.
    # For debugging one can see a verbose run with export TF_CPP_MIN_LOG_LEVEL='0' && python3 run.py
    strategy = tf.distribute.MirroredStrategy(devices=None)  # on all available GPUs  # works! # choose which GPUs shall be chosen with i.e.  export CUDA_VISIBLE_DEVICES=0,1,2,3,4 - uses then all five available GPUs
    # strategy = tf.distribute.get_strategy()  # default strategy == no strategy  # works

    with strategy.scope():
        loss = tf.keras.losses.binary_crossentropy

        inputs = tf.keras.layers.Input([3000, 16])
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = residual_block(x, n_kernels,   n_kernels*2, kernel_size=5, dilation=7)
        x = residual_block(x, n_kernels*2, n_kernels*4, kernel_size=5, dilation=7)
        x = residual_block(x, n_kernels*4, n_kernels*8, kernel_size=5, dilation=5)
        x = residual_block(x, n_kernels*8, n_kernels*8, kernel_size=5, dilation=5)
        x = residual_block(x, n_kernels*8, n_kernels*4, kernel_size=5, dilation=5)
        x = residual_block(x, n_kernels*4, n_kernels*2, kernel_size=5, dilation=5)
        x = residual_block(x, n_kernels*2, n_kernels,   kernel_size=3, dilation=3)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(.5)(x)
        x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2))(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=output, name="resnet1d_v20201002")

        model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[AUC(curve='ROC', name='roc_auc'), AUC(curve='PR', name='pr_auc'), Precision(), Recall(), mean_pred, BinaryAccuracy()])

    print(model.summary())
    return model
