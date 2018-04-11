#encoding:utf8
import pickle 
import numpy as np 
train_ratio = 0.5
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
        X = []  
        self.lbl = []
        for mod in mods:
            for snr in snrs:
                X.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  self.lbl.append((mod,snr))
        X = np.vstack(X)

        # Partition the data
        #  into training and test sets of the form we can train/test on 
        #  while keeping SNR and Mod labels handy for each
        np.random.seed(2018)
        n_examples = X.shape[0]
        n_train = int(n_examples * train_ratio)
        self.train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
        self.test_idx = list(set(range(0,n_examples))-set(self.train_idx))
        X_train = X[self.train_idx]
        X_test =  X[self.test_idx]

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
        test_SNRs = list(map(lambda x: self.lbl[x][1], self.test_idx))
        test_X_i = self.x_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = self.y_test[np.where(np.array(test_SNRs)==snr)] 

        if(num > test_X_i.shape[0]):
            raise RuntimeError('num > test_samples')
        test_idx = np.random.choice(range(0, test_X_i.shape[0]), size=num, replace=False)
        return test_X_i[test_idx], test_Y_i[test_idx]
    def next_train_batch_snr(self, num, snr):
        train_SNRs = list(map(lambda x: self.lbl[x][1], self.train_idx))
        train_X_i = self.x_train[np.where(np.array(train_SNRs)==snr)]
        train_Y_i = self.y_train[np.where(np.array(train_SNRs)==snr)] 

        if(num > train_X_i.shape[0]):
            raise RuntimeError('num > train_samples')
        train_idx = np.random.choice(range(0, train_X_i.shape[0]), size=num, replace=False)
        return train_X_i[train_idx], train_Y_i[train_idx]
