import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
from source.modals import modals
from source.utils import train
import numpy as np

if __name__ == '__main__':
    fashion_mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    X_train = X_train/255
    X_test = X_test/255
    
    model = modals(batch_size=256)
    model.set_model(2, X_train.shape[-1], y_train.shape[-1], 32)
    model.compile(optimizer = [tf.keras.optimizers.Adam(1e-3), tf.keras.optimizers.Adam(5e-4)])
        
    dir_path = 'result_run/'
    train(model, X_train[:], y_train[:], X_test[:], y_test[:], 1, path=dir_path, metric='acc')
    model.classifier.save(dir_path + 'cls_2.h5')
    model.dis.save(dir_path + 'dis_2.h5')