import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from sklearn.metrics import classification_report
from itertools import chain
from net import CustomNet
from torchvision.utils import make_grid
from torchvision import utils
import cv2
import torch.nn as nn
from dataset import train_data_reduced

matplotlib.use('TkAgg')


def visualize_single_label(data, class_number):
    for i in range(50, 100):
        img, label = data[i]
        print(label)
        print(label == class_number)
        if label == class_number:
            return img.squeeze()


def visualize_samples_image(data):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(5, 5))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def visualize_label_distribution(data, title):
    data_label_list = []

    for i in range(len(data)):
        _, label = data[i]
        data_label_list.append(label)

    data_label_list = numpy.asarray(data_label_list)

    labels, counts = numpy.unique(data_label_list, return_counts=True)

    plt.bar(labels, counts, align='center')
    plt.title(str(title))
    plt.show()


def visualize_scores(epoch_list, loss, train, test):
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(epoch_list, loss, color='r')
    ax1.set_title('Loss')

    ax2.plot(epoch_list, train, marker='o', color='g')
    ax2.set_title('Train accuracy')

    ax3.plot(epoch_list, test, marker='o', color='b')
    ax3.set_title('Test accuracy')

    plt.tight_layout()

    plt.show()


def precision_recall(preds_list, labels_list):
    preds = [preds_list[i].tolist() for i in range(len(preds_list))]
    labels = [labels_list[i].tolist() for i in range(len(labels_list))]
    preds = list(chain.from_iterable(preds))
    labels = list(chain.from_iterable(labels))
    print(classification_report(preds, labels))

    return preds, labels


def visualize_filter(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='cool'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
