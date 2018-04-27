#encoding:utf8
import pickle 
import numpy as np 
import time 
import tf_tools.normalized as normalized
import random 

train_ratio = 0.5
need_normalized = True 

threshold_variance_rate = 0.01
remove_variance = False   
# Note: This program is only tested on python3. 
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

class input_data():
    def __init__(self, file_name):
        self.n_classes = 11
        self.n_snrs = 20
        # Load the dataset ...
        #  You will need to seperately download or generate this file
        Xd = pickle.load(open(file_name,'rb'),encoding='latin1')
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
        self.mods = mods 
        X = []  
        self.lbl = []
        self.bad_idx = set()
        for mod in mods:
            for snr in snrs:
                X.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  self.lbl.append((mod,snr))
        X = np.vstack(X)
      
        # Partition the data
        #  into training and test sets of the form we can train/test on 
        #  while keeping SNR and Mod labels handy for each
        np.random.seed(int(time.time()))
        n_examples = X.shape[0]
        sample_channel = X.shape[1]
        # normalize data 
        if True == need_normalized:
            for sample_idx in range(n_examples):
                for channel_idx in range(sample_channel):
                    X[sample_idx, channel_idx, :] = \
                            normalized.my_normalized(X[sample_idx, channel_idx, :]).copy()
          # Remove some bad data
        self.bad_idx = set([])
        if remove_variance == True:
            all_var = np.zeros(X.shape[0])
            all_var_q = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                # i is index
                temp_array = np.array(X[i,0,:])
                all_var[i] = temp_array.var()

                temp_array = np.array(X[i,1,:])
                all_var_q[i] = temp_array.var()

            rank_all_var = sorted(all_var)
            threshold_idx = int(X.shape[0] * threshold_variance_rate)
            threshold_variance = rank_all_var[threshold_idx]

            rank_all_var_q = sorted(all_var_q)
            threshold_idx_q = int(X.shape[0] * threshold_variance_rate)
            threshold_variance_q = rank_all_var_q[threshold_idx_q]

            for i in range(X.shape[0]):
                if all_var[i] <= threshold_variance:
                    self.bad_idx.update(set([i]))
                if all_var_q[i] <= threshold_variance_q:
                    self.bad_idx.update(set([i]))
            print("Bad index number is ", len(self.bad_idx))
            print("threshold variance is ", threshold_variance)
            print("threshold idx is ", threshold_idx)

        n_train = int(n_examples * train_ratio)
        self.train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
        self.train_idx = list(set(self.train_idx) - self.bad_idx)
        random.shuffle(self.train_idx)

        self.test_idx = list(set(range(0,n_examples))-set(self.train_idx)- self.bad_idx)
        random.shuffle(self.test_idx)

        # need slice or deep copy?
        X_train = X[self.train_idx,:,:]
        X_test =  X[self.test_idx,:,:]

        Y_train = to_onehot(list(map(lambda x: mods.index(self.lbl[x][0]),self.train_idx)))
        Y_test = to_onehot(list(map(lambda x: mods.index(self.lbl[x][0]), self.test_idx)))

        in_shp = list(X_train.shape[1:])
        print("X_train shape is ", X_train.shape)

        self.x_test = X_test
        self.y_test = Y_test
        self.x_train = X_train
        self.y_train = Y_train

        self.train_samples = self.x_train.shape[0]
        self.test_samples = self.x_test.shape[0]
        self.samples = self.train_samples + self.test_samples

        # store different snr and different mod index
        self.test_snr_idx = {}
        self.test_mod_idx = {}
        self.train_snr_idx = {}
        self.train_mod_idx = {}
         # lbl is [idx, mod, snr]
        train_SNRs = list(map(lambda x: self.lbl[x][1], self.train_idx))
        train_MODs = list(map(lambda x: self.lbl[x][0], self.train_idx))
        
        test_SNRs = list(map(lambda x: self.lbl[x][1], self.test_idx))
        test_MODs = list(map(lambda x: self.lbl[x][0], self.test_idx))
        
        for snr in snrs:
            # return tuple
            snr_suit = np.where(np.array(train_SNRs)==snr)
            # change tuple to list to set
            self.train_snr_idx[snr]  = set(*snr_suit)

            snr_suit = np.where(np.array(test_SNRs)==snr)
            self.test_snr_idx[snr]  = set(*snr_suit)
        for mod in mods:
            mod_suit = np.where(np.array(train_MODs)==mod)
            self.train_mod_idx[mod] = set(*mod_suit)

            mod_suit = np.where(np.array(test_MODs)==mod)
            self.test_mod_idx[mod] = set(*mod_suit)

      
        
        

    def next_train_batch(self, num):
        if(num > self.train_samples):
            raise RuntimeError('num > self.train_samples')
        train_idx = np.random.choice(range(0, self.train_samples), size=num, replace=False)
        return self.x_train[train_idx], self.y_train[train_idx]
        
    def next_test_batch(self, num):
        if(num > self.test_samples):
            raise RuntimeError('num > self.test_samples')
        test_idx = np.random.choice(range(0, self.test_samples), size=num, replace=False)
        return self.x_test[test_idx], self.y_test[test_idx]
       
    def next_test_batch_snr(self, num, snr):
        idx = list(self.test_snr_idx[snr])
        test_X_i = self.x_test[idx]
        test_Y_i = self.y_test[idx] 

        if(num > test_X_i.shape[0]):
            raise RuntimeError('num > test_samples')
        test_idx = np.random.choice(range(0, test_X_i.shape[0]), size=num, replace=False)
        return test_X_i[test_idx], test_Y_i[test_idx]
    def next_train_batch_snr(self, num, snr):
        idx = list(self.train_snr_idx[snr])
        train_X_i = self.x_train[idx]
        train_Y_i = self.y_train[idx] 

        if(num > train_X_i.shape[0]):
            raise RuntimeError('num > train_samples')
        train_idx = np.random.choice(range(0, train_X_i.shape[0]), size=num, replace=False)
        return train_X_i[train_idx], train_Y_i[train_idx]
    def next_train_batch_snr_mod(self, num, snr, mod):

        idx = list(self.train_snr_idx[snr] & self.train_mod_idx[mod])
        train_X_i = self.x_train[ idx]
        train_Y_i = self.y_train[ idx] 

        if(num > train_X_i.shape[0]):
            raise RuntimeError('num > train_samples')
        train_idx = np.random.choice(range(0, train_X_i.shape[0]), size=num, replace=False)
        #print('train idx is ', train_idx)
        return train_X_i[train_idx], train_Y_i[train_idx]
