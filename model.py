import os
import torch
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from network import network
from data import build_dataloader_CY101
from torch.nn import functional as F

class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        train_dataloader, valid_dataloader = build_dataloader_CY101(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader}
        self.net = network(opt=self.opt, channels=opt.channels, height=self.opt.height, width=self.opt.width)

        self.net.to(self.device)
        self.bce_loss = nn.BCELoss()
        print('# parameters:', sum(param.numel() for param in self.net.parameters()))
        if self.opt.pretrained_model:
            self.load_weight()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate, weight_decay=1e-4)

    def train_epoch(self, epoch):
        print("--------------------start training epoch %2d--------------------" % epoch)
        for iter_, (images, behaviors) in enumerate(self.dataloader['train']):
            self.net.zero_grad()

            images = images.permute([1, 0, 2, 3, 4]).unbind(0)

            preds = self.net(images)

            loss = self.bce_loss(preds, behaviors)
            loss.backward()
            self.optimizer.step()
            if iter_ % self.opt.print_interval == 0:
                print("training epoch: %3d, iterations: %3d/%3d loss: %6f " %
                      (epoch, iter_, len(self.dataloader['train'].dataset)//self.opt.batch_size, loss))

    def train(self):
        interval = self.opt.epochs // 5
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            if epoch_i % interval == 0:
                self.evaluate(epoch_i)
                self.save_weight(epoch_i)

    def evaluate(self, epoch):
        # loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for iter_, (images, behaviors) in enumerate(self.dataloader['valid']):
                self.net.zero_grad()
                images = images.permute([1, 0, 2, 3, 4]).unbind(0)
                preds = self.net(images)
                _, predicted = torch.max(preds.data, 1)
                _, labels = torch.max(behaviors.data, 1)
                total += behaviors.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self, path=None):
        if path:
            self.net.load_state_dict(torch.load(path))
        elif self.opt.pretrained_model:
            self.net.load_state_dict(torch.load(self.opt.pretrained_model, map_location=torch.device('cpu')))
