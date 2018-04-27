import os 
import tensorflow as tf 

def select_gpu(gpu_number):
    # select gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
def set_memory_limit(percentage):
    config = tf.ConfigProto() 
    # 占用显存百分比
    config.gpu_options.per_process_gpu_memory_fraction = percentage
    session = tf.Session(config=config)
    session.close()
def enable_memory_growth():
    # 按需占用GPU
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    session = tf.Session(config=config)
    session.close()
def disable_memory_growth():
    # 按需占用GPU
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=False   
    session = tf.Session(config=config)
    session.close()