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
    import seaborn as sns
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    iris = sns.load_dataset('iris')
    
    X = iris.iloc[:,0:4].values
    y = iris.iloc[:,4].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size=0.2, 
                                                        random_state=1) 
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    X_test /= np.max(X_train, axis=0)
    X_train /= np.max(X_train, axis=0)
    
    model = modals(batch_size=32)
    model.set_model(2, X_train.shape[-1], y_train.shape[-1], 32)
    model.compile(optimizer = [tf.keras.optimizers.Adam(1e-3), tf.keras.optimizers.Adam(5e-4)])
        
    dir_path = 'result_run_iris/'
    train(model, X_train[:], y_train[:], X_test[:], y_test[:], 0, path=dir_path, metric='acc')
    model.classifier.save(dir_path + 'cls.h5')
    model.dis.save(dir_path + 'dis.h5')