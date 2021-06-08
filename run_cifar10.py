import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
from source.modals import modals_cov as modals
from source.utils import train
import numpy as np

def make_cov_model(input_dim, output_dim, flag=True):
    inputs = tf.keras.Input(shape=(input_dim))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)    
    if flag:
        outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(output_dim, activation='relu')(x)
    return tf.keras.Model(inputs, outputs, name='cifar10')

if __name__ == '__main__':
    reduce_cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = reduce_cifar10.load_data()
    
    # 76% -> 86%
    # need conv net
    
    y_train = np.squeeze(np.eye(10)[y_train])
    y_test = np.squeeze(np.eye(10)[y_test])
    
    print(y_train.shape, X_train.shape)

    #normalization
    X_train = X_train/255
    X_test = X_test/255
    
    # base model : 80.6%
#     model = make_cov_model(X_train.shape[1:], y_train.shape[-1])
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))
    
    model = modals(batch_size=128)
    model.set_model(1, X_train.shape[1:], y_train.shape[-1], 32)
    
    # model change
#     model.fe = make_cov_model(X_train.shape[1:], 32, False)
#     inputs = tf.keras.Input(shape=X_train.shape[1:])
#     outputs = model.dl(model.fe(inputs))
#     model.classifier = tf.keras.Model(inputs=inputs, outputs=outputs, name='classifier')
#     print(model.fe(X_train[:3]).shape)
    
    model.compile(optimizer = [tf.keras.optimizers.Adam(1e-3), tf.keras.optimizers.Adam(5e-4)])
    
    dir_path = 'result_run_cifar10/'
    train(model, X_train[:], y_train[:], X_test[:], y_test[:], 0, path=dir_path, metric='acc')
    model.classifier.save(dir_path + 'cls.h5')
    model.dis.save(dir_path + 'dis.h5')