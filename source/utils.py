from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, auc
pca = PCA(n_components=2)

def new_param(param):
    param_list = [['inter', 'exp', 'noise', 'diff'], # method
                    np.arange(0, 1.1, 0.1).tolist(), # augmentation probability
                    np.arange(0, 1, 0.1).tolist(), # augmentation rambda
                 ]
    
    for i in range(len(param)):
        if np.random.rand() < 0.2: # Resample
            # resample param
            index = np.arange(len(param_list[i]))
            np.random.shuffle(index)
            param[i] = param_list[i][index[0]]
        else: # Perturb
            amt = np.random.randint(4)
            if np.random.rand() < 0.5:
                param[i] = param_list[i][int(min(np.where(np.array(param_list[i]) == param[i])[0][0] + amt, len(param_list[i])-1))]
            else:
                param[i] = param_list[i][int(max(np.where(np.array(param_list[i]) == param[i])[0][0] - amt, 0))]
    return param

def exploit(param_list, loss_list):
    if np.random.rand() < 0.2: # Binary tournament random selection
        for i in range(len(param_list)):
            sample_index = np.random.randint(len(param_list))
            if loss_list[i] > loss_list[sample_index]:
                param_list[i] = param_list[sample_index].copy()
    else: # Truncation selection bottom 20% -> top 20% 
        ranking_index = np.argsort(loss_list)
        len_20 = int(max(len(loss_list) * 0.2, 1))
        for i in range(len_20):
            sample_index = np.random.randint(len_20)
            param_list[ranking_index[-(i+1)]] = param_list[ranking_index[sample_index]].copy()
            
    return param_list

def explore(param_list):
    for i in range(len(param_list)):
        param_list[i] = new_param(param_list[i]).copy()
    return param_list

# train 
def train(model, X_train, y_train, X_val, y_val, fold, path='result/', epoch=100, new_param_term=3, search_model=16, metric='AUC'):
    def make_batch():
        y_num_train = np.argmax(y_train, axis=1)
        x_data_dict = {}
        total_size = len(X_train)
        each_sample_per_epoch= {}
        for i in np.unique(y_num_train):
            x_data_dict[i] = X_train[y_num_train == i]
        
        each_sample = np.round(total_size/model.batch_size)
        for key in x_data_dict.keys():
            each_sample_per_epoch[key] = int(np.round(len(x_data_dict[key])/each_sample - 0.5))#int(np.round(len(x_data_dict[key])/each_sample))
        
        i = 0
        while True:
            i += 1
            if i >= each_sample:
                i = 0
            X_batch = []
            y_batch = []
            for key in x_data_dict.keys():
                gap = each_sample_per_epoch[key]
                X_batch += x_data_dict[key][i * gap : (i + 1) * gap].tolist()
                y_batch += [key] * len(x_data_dict[key][i * gap : (i + 1) * gap])
            yield np.array(X_batch), np.eye(len(each_sample_per_epoch))[y_batch]#np.array(y_batch)
            
        
    #print(make_batch())
    count = 0
    
    step_total = len(X_train)/model.batch_size
    
    param_list = []
    for i in range(search_model):
        param_list.append(new_param(model.param))
    param_list = explore(param_list)
    
    log_data = []
    log_column = ['epoch', 'index', 'method', 'percent', 'lambda', 'loss', 'val_loss', 'classifier_loss', 'triplet_loss', 'adversarial_loss', 'discriminator_loss', 'auc']
    
    for i in range(0, epoch, 3):
        # clip pre param
        pre_param = model.param.copy()
        
        # save model status
        base_classifier_weight = model.classifier.get_weights()
        base_discriminator_weight = model.dis.get_weights()
        
        # best search data
        best_loss = 99999
        best_param = []
        best_classifier_weight = []
        best_discriminator_weight = []
        loss_list = []
        model_weights = []
        
        for j in range(search_model):
            # new param set
            model.param = param_list[j]
            
            # reset weights
            if len(model_weights) == j:
                model.classifier.set_weights(base_classifier_weight)
                model.dis.set_weights(base_discriminator_weight)
                model_weights.append([base_classifier_weight, base_discriminator_weight])
            else:
                model.classifier.set_weights(model_weights[j][0])
                model.dis.set_weights(model_weights[j][1])
            
            # training model new_param_term
            
            for ii in range(new_param_term):    
                loss = 0
                loss_dict = {'classifier_loss': 0, 'triplet_loss': 0, 'adversarial_loss': 0, 'discriminator_loss': 0, 'loss_M': 0}
                for k, batch_data in tqdm(enumerate(make_batch()), total=int(np.round(step_total))):
                    loss_dict_tmp = model.train_step(batch_data)
                    loss_dict = {key:loss_dict[key] + loss_dict_tmp[key] for key in loss_dict.keys()} 
                    loss = loss_dict['loss_M'].numpy()
                    if k >= step_total:
#                     if True: # test
                        break
                loss /= np.round(step_total)
                loss_dict = {key:loss_dict[key]/np.round(step_total) for key in loss_dict.keys()}
                loss_string = ''
                for key in loss_dict.keys():
                    loss_string += key + ' : ' + str(loss_dict[key].numpy()) + ', '
                print(str(j) + 'model, ' + 'epoch', i + ii, loss_string, 'total_loss :', loss, 'param :', model.param)
                
            model_weights[j] = [model.classifier.get_weights(), model.dis.get_weights()]

            # loss_list.append(loss) # train_loss
            val_loss = 0
            loss_dict_tmp = model.test_step((X_val, y_val))
#             val_loss += np.sum([loss_dict_tmp[key].numpy() for key in loss_dict_tmp.keys()])
            val_loss = loss_dict_tmp['loss_M'].numpy()
            y_pred = model.classifier(X_val)
            if metric == 'AUC':
                fpr, tpr, th = roc_curve(y_val[:, 0], y_pred[:, 0])
                acc = auc(fpr, tpr)
            elif metric == 'acc':
                acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
            print(val_loss, acc)
            loss_list.append(val_loss) # validation_loss
            
            
            # log write
            # ['epoch', 'index', 'method', 'percent', 'lambda', 'loss', 'val_loss', 'classifier_loss', 'triplet_loss', 'adversarial_loss', 'discriminator_loss']
            log_data.append([i, j, param_list[j][0], param_list[j][1], param_list[j][2], 
                             loss, val_loss, loss_dict_tmp['classifier_loss'].numpy(), loss_dict_tmp['triplet_loss'].numpy(), 
                             loss_dict_tmp['adversarial_loss'].numpy(), loss_dict_tmp['discriminator_loss'].numpy(), acc])
            pd.DataFrame(log_data, columns=log_column).to_csv(path + 'fold' + str(fold) + '_log.csv', index=False)
            
            if best_loss >= val_loss: # validation_loss
                best_loss = val_loss
                best_classifier_weight = model.classifier.get_weights()
                best_discriminator_weight = model.dis.get_weights()
        
        # param setting 
        param_list = exploit(param_list, loss_list)
        param_list = explore(param_list)
        
        model.classifier.set_weights(best_classifier_weight)
        model.dis.set_weights(best_discriminator_weight)
        print('best loss :', best_loss, 'search end')
        y_pred = model.classifier(X_val)
        print(np.argmax(y_pred, axis=1), np.argmax(y_val, axis=1))
        if metric == 'AUC':
            fpr, tpr, th = roc_curve(y_val[:, 0], y_pred[:, 0])
            rr = auc(fpr, tpr)
        elif metric == 'acc':
            rr = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
            
        plt.figure(figsize=(20, 10))
        # random 1000 sample 
        # train plot
        ax1 = plt.subplot(121)
        draw_result(ax1, X_train, y_train, model)
        
        # validation plot
        ax2 = plt.subplot(122)
        draw_result(ax2, X_val, y_val, model, False)

        plt.legend()
        plt.title('best loss :' + str(best_loss) + 'search end,' + metric + ' :' + str(rr))
        plt.savefig(path + str(fold) + 'fold_' + str(i+3) + 'epoch.png')
        
        print(accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1)))

def draw_result(ax, X, y, model, flag=True):
    index = np.arange(y.shape[0])
    np.random.shuffle(index)
    index = index[:1000]
    z = model.fe(X[index])
    if flag:
        pca.fit(z)
    z_t = pca.transform(z)
    
    for i in range(y.shape[-1]):
        ax.scatter(z_t[np.argmax(y[index], axis=1) == i][:, 0], z_t[np.argmax(y[index], axis=1) == i][:, 1], alpha=0.3, label=str(i))
    return pca
    
