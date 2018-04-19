#coding:utf-8
import numpy as np 


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
def wgn_dimension2(x, snr):
    temp = np.zeros(x.shape)
    for i in range(x.shape[0]):
        temp[i,:] = wgn(x[i,:], snr)
    return temp 
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    t = np.linspace(0, 2*np.pi, 100)
    x = np.sin(t)
    n = wgn(x, 40)
    xn = x+n 
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(t,x)
    plt.subplot(1,2,2)
    plt.plot(t,xn)
    print("min(xn)=%f,max(xn)=%f" %(min(xn),max(xn)))

    plt.savefig('./test_signal_process')

    # test dimension2
    x = np.zeros([3,3])
    n = wgn_dimension2(x,40)
    xn = x + n