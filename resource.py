import os 
import tensorflow as tf 
_gpu_number = 0
def select_gpu(gpu_number):
    # select gpu
    global _gpu_number
    _gpu_number = gpu_number
def set_memory_limit(percentage):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_number)
    config = tf.ConfigProto() 
    # 占用显存百分比
    config.gpu_options.per_process_gpu_memory_fraction = percentage
    session = tf.Session(config=config)
    session.close()
def enable_memory_growth():
    # 按需占用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_number)
    print('_gpu_number is', _gpu_number)
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    session = tf.Session(config=config)
    session.close()
def disable_memory_growth():
    # 按需占用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_number)
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=False   
    session = tf.Session(config=config)
    session.close()