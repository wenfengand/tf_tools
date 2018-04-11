import matplotlib.pyplot as plt 
import numpy as np 
import random 

def plot_confusion_matrix(predicted, labels, classes, file_name, title):
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,predicted.shape[0]):
        j = list(labels[i,:]).index(1)
        k = int(np.argmax(predicted[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plt.imshow(confnorm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)
if __name__ == '__main__':
    # test it
    batch_size = 100
    classes = [1, 2, 3, 4, 5, 6, 7, 8]
    length_class = len(classes)
    predicted_label = np.zeros([batch_size, length_class])
    real_label = np.zeros([batch_size, length_class])

    for i in range(batch_size):
        random_idx = random.randint(0, length_class - 1)
        predicted_label[i, random_idx] = 1
        random_idx = random.randint(0, length_class - 1)
        real_label[i, random_idx] = 1
    # assign real label to be the same with predicted label, accuracy is 100%
    # real_label = predicted_label
    plot_confusion_matrix(predicted_label, real_label, classes, 'test.png', 'test' )    
