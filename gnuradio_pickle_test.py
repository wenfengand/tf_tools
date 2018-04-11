#encoding:utf8
import gnuradio_pickle
file_name = '/home/wangwenfeng/dataset/GNUradio/RML2016.10a_dict.dat'
gnu = gnuradio_pickle.input_data(file_name)
test_x,test_y = gnu.next_test_batch_snr(100, 0)
train_x,train_y = gnu.next_train_batch_snr(100, 0)
print("testx shape is",test_x.shape)
print("train_x shape is", train_x.shape)