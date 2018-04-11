#encoding:utf8
import gnuradio_pickle
import matplotlib.pyplot as plt 
import numpy as np 
file_name = '/home/wangwenfeng/dataset/GNUradio/RML2016.10a_dict.dat'
gnu = gnuradio_pickle.input_data(file_name)
test_x,test_y = gnu.next_test_batch_snr(100, 0)
train_x,train_y = gnu.next_train_batch_snr(100, 0)
print("testx shape is",test_x.shape)
print("train_x shape is", train_x.shape)
targetx,targety = gnu.next_train_batch_snr_mod(10, 18, gnu.mods[0])
print('mod is %s, targetx shape is' %gnu.mods[0], targetx.shape)

# plot different mod
# to eliminate runtime error: no dispaly via ssh
plt.switch_backend('agg')
plt.figure()
mods = gnu.mods
pic_idx = 1
show_ticks = True
for mod in mods:
    plt.subplot(3,4,pic_idx)
    print('mod is ', mod)
    y,_= gnu.next_train_batch_snr_mod(1,18,mod)
    y = y[:,0,:]
    y = y.reshape(128)
    x = np.linspace(0, 127, 128)
    plt.plot(x,y)
    plt.title(mod)
    
    # don't show x y ticks
    if False==show_ticks:
        plt.xticks([])
        plt.yticks([])
    pic_idx = pic_idx + 1
plt.savefig('plot_different_mod.png')

