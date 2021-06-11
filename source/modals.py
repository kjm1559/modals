import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

# alpha = 1, beta = 0.03, margin = {0.5, 1, 2, 4, 8}

class modals(tf.keras.Model):
    def make_feature_extractor(self, layers, input_dim, hidden_size):
        inputs = tf.keras.Input(shape=(None, input_dim))
        x = inputs
        for i in range(layers - 1):
            x = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(x)
        outputs = tf.keras.layers.LSTM(hidden_size)(x)
        
        return tf.keras.Model(inputs, outputs, name='feature_extractor')
    
    def make_dense_layer(self, hidden_size, output_dim):
        inputs = tf.keras.Input(shape=(hidden_size))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name='dense_layer')
    
    def make_discriminator(self, hidden_size):
        inputs = tf.keras.Input(shape=(hidden_size))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Activation('relu')(x)
#         x = tf.keras.layers.Dropout(0.2)(x)
#         x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs, name='discriminator')
    
    def cal_latent(self, X, training=False):
        return self.fe(X, training=training)
    
    def __cosine_distance(self, a, b):
        return 1 - K.sum(a * b) / (K.sqrt(K.sum(K.square(a))) * K.sqrt(K.sum(K.square(b))))
    
    def tf_triplet_selector(self, z, y, training=False):
        z_inclass = []
        z_outclass = []
        for i in range(y.shape[0]):
            idx_in = np.random.choice(np.where(np.argmax(y, axis=1) == np.argmax(y[i]))[0], 1)[0]
            z_inclass.append(z[idx_in])
            
            idx_out = np.random.choice(np.where(np.argmax(y, axis=1) != np.argmax(y[i]))[0], 1)[0]
            z_outclass.append(z[idx_out])
            
        z_inclass = tf.stack(z_inclass)
        z_outclass = tf.stack(z_outclass)  
        return K.mean((self.__cosine_distance(z, z_inclass)) - (self.__cosine_distance(z, z_outclass)) + self.gamma)
                                 
                                 
        
    def __split_data(self):
        class_num_y = np.argmax(self.y_train, axis=1)
        self.X_data_dict = {}
        unique_class = np.unique(class_num_y)
        for i in unique_class:
            index = np.where(class_num_y == i)[0]
            self.X_data_dict[i] = []
            for j in index:
                self.X_data_dict[i].append(self.X_train[j])
    
    def __get_inclass_z(self, y):
        return self.z_data_dict[y]
    
    def __get_outclass_z(self, y):
        classes = [x for x in self.z_data_dict.keys() if x != y]
        return self.z_data_dict[classes[np.random.randint(len(classes))]]
                
    def cal_hard_interpolation(self, z, y):
        inclass_z = self.__get_inclass_z(y)
        center = np.mean(inclass_z, axis=0)
        # each distanse
        distanse = np.sqrt(np.sum(np.square(center - inclass_z), axis=1))
        index_sort = np.argsort(distanse)
        top_z = inclass_z.numpy()[index_sort[-max(int(len(inclass_z) * 0.05), 1):]] # 5% select
        z_dist = np.sqrt(np.sum(np.square(z - top_z), axis=1))
        index_sort = np.argsort(z_dist)
        return z + self.param[2]*(top_z[index_sort[0]] - z)
    
    def cal_hard_expolation(self, z, y):
        mu = np.mean(self.__get_inclass_z(y), axis=0)
        return z - self.param[2]*(z - mu)
    
    def cal_gaussian_noise(self, z, y):
        sigma = np.std(self.__get_inclass_z(y))
        return z + self.param[2] * np.random.normal(0, sigma, z.shape)
    
    def cal_difference(self, z, y):
        # select 2 sample from same calss
        X_tmp = self.__get_inclass_z(y)
        shuffle_index = np.arange(len(X_tmp))
        np.random.shuffle(shuffle_index)
        return z + self.param[2]*(X_tmp[shuffle_index[0]] - X_tmp[shuffle_index[1]])
    
    def get_classification_loss(self, z, y, training=False):
        z_hat = tf.unstack(z)
        for i in range(len(z_hat)):
            target_z = z_hat[i]
            if np.random.rand() < self.param[1]:
                if 'inter' == self.param[0]:
                    z_hat[i] = self.cal_hard_interpolation(target_z, np.argmax(y[i]))
                elif 'exp' == self.param[0]:
                    z_hat[i] = self.cal_hard_expolation(target_z, np.argmax(y[i]))
                elif 'noise' == self.param[0]:
                    z_hat[i] = self.cal_gaussian_noise(target_z, np.argmax(y[i]))
                elif 'diff' == self.param[0]:
                    z_hat[i] = self.cal_difference(target_z, np.argmax(y[i]))

                if len(z_hat[i].shape) != 1:
                    print(z_hat[i].shape, z[i].shape)#z_hat[i])

        z_hat = tf.stack(z_hat)
        classes = self.dl(z_hat, training=training)
        
#         return K.mean(tf.keras.losses.categorical_crossentropy(np.array(new_y), classes))
        return -K.mean((K.log(K.sum(y * classes, axis=1) + 1e-8)))
    
    def set_param(self, param=['inter', 0.1, 0.1]):
        # param = [op, prob, mag]
        self.param = param
        
    def set_model(self, layers, input_dim, output_dim, hidden_size):
        self.fe = self.make_feature_extractor(layers, input_dim, hidden_size)
        self.dl = self.make_dense_layer(hidden_size, output_dim)
        
        # define classifier 
        inputs = tf.keras.Input(shape=(None, input_dim,))
        outputs = self.dl(self.fe(inputs))
        self.classifier = tf.keras.Model(inputs=inputs, outputs=outputs, name='classifier')
        self.dis = self.make_discriminator(hidden_size)
        
    def __init__(self, gamma=1, batch_size=256):
        super(modals, self).__init__()
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.alpha = 1
        self.beta = 0.03
        
        self.set_param() # set initial paramater
            
    def compile(self, optimizer=None, **arg):
        super(modals, self).compile(optimizer, loss='mse')
        self.M_optimizer = optimizer[0]
        self.D_optimizer = optimizer[1]
    
    def train_step(self, batch_data):
        X, y = batch_data
        self.X_train = X
        self.y_train = y
        self.__split_data()
        self.z_data_dict = {}
        
        if X.shape[0] != None:
            with tf.GradientTape() as tape:
                z = self.fe(X, training=True)
                for key in self.X_data_dict.keys():
                    self.z_data_dict[key] = z[np.argmax(y, axis=1) == key]#self.cal_latent(self.X_data_dict[key], training=True)
                classification_loss = self.get_classification_loss(z, y, training=True)
                triplet_loss = self.tf_triplet_selector(z, y, training=True)
                adversarial_loss = -K.mean(K.log(self.dis(z, training=False) + 1e-8))

                loss_M = classification_loss + self.alpha * adversarial_loss + self.beta * triplet_loss

            grads_M = tape.gradient(loss_M, self.classifier.trainable_variables)
            self.M_optimizer.apply_gradients(zip(grads_M, self.classifier.trainable_variables))

            with tf.GradientTape() as tape:
                z = self.fe(X, training=True)
#                 norm_z = tf.random.normal(z.shape, mean=0.0, stddev=np.std(z, axis=0), dtype=tf.float32)
                norm_z = tf.random.normal(z.shape, mean=0.0, stddev=1, dtype=tf.float32)
                noise_d = self.dis(norm_z, training=True)
                z_d = self.dis(z, training=True)

                loss_D = -K.mean(K.log(noise_d + 1e-8) + K.log(1 - z_d + 1e-8))

            grads_D = tape.gradient(loss_D, self.dis.trainable_variables)
            self.D_optimizer.apply_gradients(zip(grads_D, self.dis.trainable_variables))
            
            return {
                'classifier_loss': classification_loss,
                'triplet_loss': triplet_loss,
                'adversarial_loss': adversarial_loss,
                'loss_M': loss_M,
                'discriminator_loss': loss_D,
            }
        else:
            print('initial')
            print(X, y)
            return {
                'classifier_loss': -1,
                'triplet_loss': -1,
                'adversarial_loss': -1,
                'loss_M': -1,
                'discriminator_loss': -1,
            }
        
    
    def test_step(self, batch_data):
        X, y = batch_data
        self.X_train = X
        self.y_train = y
        self.__split_data()
        self.z_data_dict = {}
        if X.shape[0] != None:
            z = self.cal_latent(X, training=False)
            for key in self.X_data_dict.keys():
                self.z_data_dict[key] = z[np.argmax(y, axis=1) == key]#self.cal_latent(self.X_data_dict[key], training=True)
            classification_loss = self.get_classification_loss(z, y)
            triplet_loss = self.tf_triplet_selector(z, y, training=True)
            adversarial_loss = -K.mean(K.log(self.dis(z)))

            loss_M = classification_loss + self.alpha * adversarial_loss + self.beta * tf.cast(triplet_loss, 'float32')

            z = np.concatenate([self.z_data_dict[key] for key in self.z_data_dict.keys()], axis=0)
            norm_z = tf.random.normal(z.shape, mean=0.0, stddev=np.std(z, axis=0), dtype=tf.float32)

            loss_D = -K.mean(K.log(self.dis(norm_z, training=False)) + K.log(1 - self.dis(z, training=False)))
            
            return {
                'classifier_loss': classification_loss,
                'triplet_loss': triplet_loss,
                'adversarial_loss': adversarial_loss,
                'loss_M': loss_M,
                'discriminator_loss': loss_D,
            }
        else:
            print('initial')
            print(X, y)
            return {
                'classifier_loss': -1,
                'triplet_loss': -1,
                'adversarial_loss': -1,
                'loss_M': -1,
                'discriminator_loss': -1,
            }     
        
        
class modals_cov(modals):
    def make_feature_extractor(self, layers, input_dim, hidden_size):
        inputs = tf.keras.Input(shape=(input_dim))
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)    
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(hidden_size, activation='linear')(x)
        return tf.keras.Model(inputs, outputs, name='cifar10')
    
    def set_model(self, layers, input_dim, output_dim, hidden_size):
        self.fe = self.make_feature_extractor(layers, input_dim, hidden_size)
        self.dl = self.make_dense_layer(hidden_size, output_dim)
        
        # define classifier 
        inputs = tf.keras.Input(shape=input_dim)
        outputs = self.dl(self.fe(inputs))
        self.classifier = tf.keras.Model(inputs=inputs, outputs=outputs, name='classifier')
        self.fe.summary()
        self.classifier.summary()
        self.dis = self.make_discriminator(hidden_size)
