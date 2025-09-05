from __future__ import print_function, absolute_import
import time
import torch
import torch.nn.functional as F
from .utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, encoder, memory=None,memory_cam=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_cluster2camera = memory_cam

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, cams, indexes = self._parse_data(inputs)

            loss_hybrid = 0
            loss_cam = 0
            # forward
            f_out = self._forward(inputs)
            loss_hybrid += self.memory(f_out, labels)
            loss_cam += self.memory_cluster2camera(f_out, labels ,cams)
            total_loss = loss_hybrid + 0.3 * loss_cam
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.update(total_loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), cids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
