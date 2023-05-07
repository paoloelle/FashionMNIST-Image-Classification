import torchvision.models
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.models.resnet import BasicBlock
from dataset import *
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch
from net import CustomNet
from utils import *
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import numpy as np


class Solver():
    def __init__(self, args):

        self.train_data = train_data_reduced
        self.test_data = test_data_reduced

        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=8,
                                       shuffle=True, drop_last=True
                                       )

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.customNet:
            self.net = CustomNet().to(self.device)

        else:  # download and adapt ResNet-18

            self.net = torchvision.models.resnet18(pretrained=True).to(self.device)
            for param in self.net.parameters():
                param.requires_grad = False

            # Parameters of newly constructed modules have requires_grad=True by default
            self.net.conv1 = nn.Conv2d(in_channels=1,
                                       out_channels=64,
                                       kernel_size=7,
                                       stride=2,
                                       padding=3,
                                       bias=False).to(self.device)
            # original ResNet layer: (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            self.net.fc = nn.Linear(self.net.fc.in_features, 10).to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr=args.lr)
        self.args = args

    def fit(self):

        print(self.net)

        # lists for trend of scores visualization
        # loss_list = []
        # epoch_list = []
        # train_accuracy_list = []
        # test_accuracy_list = []

        args = self.args

        for epoch in range(args.max_epochs):

            self.net.train()

            for step, inputs in enumerate(self.train_loader):
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)

                # forward
                pred = self.net(images)
                loss = self.loss_fn(pred, labels)

                # loss and optimize
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                print("print:", self.net.conv1.weight.data[0, 0, :])

            if (epoch + 1) % args.print_every == 0:
                # evaluation
                train_acc, preds_train, labels_train = self.evaluate(self.train_data)
                test_acc, preds_test, labels_test = self.evaluate(self.test_data)

                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                      format(epoch + 1, args.max_epochs, loss.item(), train_acc, test_acc))

            # loss_list.append(loss.item())
            # epoch_list.append(epoch + 1)
            # train_accuracy_list.append(train_acc)
            # test_accuracy_list.append(test_acc)

        ###### PRINT FEATURE MAP ###########
        
        # not the best place to do this 

        # no_of_layers = 0
        # conv_layers = []

        # self.net.to(torch.device('cpu'))

        # model_children = list(self.net.children())
        # print(model_children)

        # for child in model_children:
        #     if type(child) == nn.Conv2d:
        #         no_of_layers += 1
        #         conv_layers.append(child)
        #     elif type(child) == nn.Sequential:
        #         for layer in child.children():
        #             if type(layer) == nn.Conv2d:
        #                 no_of_layers += 1
        #                 conv_layers.append(layer)

        # print(len(conv_layers))
        #
        # print(labels[0])
        #
        # print(images[0].size())
        # image = images[0].to(torch.device('cpu'))
        # image = image.unsqueeze(1)
        # print(image.size())
        #
        # result = [conv_layers[0](image)]
        #
        # for i in range(1, len(conv_layers)):
        #     result.append(conv_layers[i](result[-1]))
        # output = result
        #
        # boh = list()
        #
        # for num_layer in range(len(output)):
        #     layer_viz = output[num_layer][0, :, :, :]
        #     layer_viz = layer_viz.data
        #     filter = layer_viz[10]  # 10 feature map
        #     boh.append(filter)
        #
        # fig = plt.figure(figsize=(10, 10))
        # columns = 10
        # rows = 1
        # fig.add_subplot(rows, columns, 1)
        # plt.axis('off')
        # plt.imshow(image.squeeze(), cmap="gray")
        # for i in range(3):
        #     img = boh[i]
        #     fig.add_subplot(rows, columns, i + 2)
        #     # plt.title('Feature map layer' + str(i))
        #     plt.axis('off')
        #     plt.imshow(img, cmap="YlOrRd")
        # plt.show()

        #### END PRINT FEATURE MAP


        # FILTER VISUALIZATION

        # also here definitely not the best place

        # visualize_filter(self.net.conv1[0].weight.data.to(torch.device('cpu')), ch=0, allkernels=False)
        # visualize_filter(self.net.conv2[0].weight.data.to(torch.device('cpu')), ch=0, allkernels=False)
        # visualize_filter(self.net.conv3[0].weight.data.to(torch.device('cpu')), ch=0, allkernels=False)



        # SCORES

        # visualize_scores(epoch_list, loss_list, train_accuracy_list, test_accuracy_list)

        # precision_recall(preds_train, labels_train)
        # precision_recall(preds_test, labels_test)

        # pred_cf, label_cf = precision_recall(preds_test, labels_test)

        # confusion matrix
        # cf_matrix = confusion_matrix(label_cf, pred_cf)
        # plot_confusion_matrix(cf_matrix, train_data.classes)
        # plt.show()

    def evaluate(self, data):

        predictions_list = []
        labels_list = []

        args = self.args
        loader = DataLoader(data,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=False)

        self.net.eval()

        num_correct, num_total = 0, 0

        with torch.no_grad():
            for inputs in loader:
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)

                labels_list.append(labels)

                outputs = self.net(images)
                _, preds = torch.max(outputs.detach(), 1)

                predictions_list.append(preds)

                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total, predictions_list, labels_list
