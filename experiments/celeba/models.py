import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import chain


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_base = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.weight_network = nn.Sequential(
            nn.Linear(512, 40),
            nn.ReLU(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.weights = nn.Parameter(torch.ones(40))
        
        self.softmax_temperature = 3.0
        # self.lower_limit = 0.01
        # self.upper_limit = 0.9
        
        # Prediction head
        self.out_layer = nn.ModuleList([nn.Linear(512, 1) for _ in range(40)])

    def forward(self, x, task=None, return_representation=False):
        h = self.shared_base(x)
        # weights = self.weight_network(h.mean(dim=0))
        # weights = torch.nn.functional.softmax(weights / self.softmax_temperature, dim=-1)
        # weights = torch.clamp(weights, min=self.lower_limit, max=self.upper_limit)
        # weights = weights / weights.sum(dim=-1, keepdim=True)
        if task is None:
            y = [torch.sigmoid(self.out_layer[task](h)) for task in range(40)]
        else:
            y = torch.sigmoid(self.out_layer[task](h))

        if return_representation:
            return y, torch.sigmoid(self.weights), h
        else:
            return y, torch.sigmoid(self.weights)
            # return y, weights

    def shared_parameters(self):
        return (p for p in self.shared_base.parameters())

    def task_specific_parameters(self):
        return_list = []
        for task in range(40):
            return_list += [p for p in self.out_layer[task].parameters()]
        return return_list

    def last_shared_parameters(self):
        return []
    
    def model_parameters(self):
        return (p for n, p in self.named_parameters() if "weights" not in n)
