import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from utils import read_graph, feature_calculator, adjacency_opposite_calculator

class AttentionWalkLayer(torch.nn.Module):

    def __init__(self, args, shapes):
        super(AttentionWalkLayer, self).__init__()
        self.args = args
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the parameters.
        """
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], self.args.dimensions))
        self.right_factors = torch.nn.Parameter(torch.Tensor(self.args.dimensions,self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0],1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.xavier_normal_(self.left_factors)
        torch.nn.init.xavier_normal_(self.right_factors)
        torch.nn.init.xavier_normal_(self.attention)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        if self.args.geometric == True:
            print("Geometric")
        else:
            self.attention_probs = torch.nn.functional.softmax(self.attention, dim = 0)
        print(self.attention_probs)

        weighted_target_tensor = weighted_target_tensor * self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_matrix = torch.sum(weighted_target_tensor, dim=0).view(self.shapes[1],self.shapes[2])
        loss_on_target = - weighted_target_matrix * torch.log(torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_opposite = - adjacency_opposite * torch.log(1-torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_on_matrices = (loss_on_target + loss_opposite).norm(1)
        loss_on_regularization = self.args.lamb* (self.attention.norm(2)**2)
        loss = loss_on_matrices +  loss_on_regularization
        print(loss)
        return loss
        

class AttentionWalkTrainer(object):
    '''
    '''
    def __init__(self, args):
        self.args = args
        self.graph = read_graph(self.args.edge_path)
        self.initialize_model_and_features()

    def initialize_model_and_features(self):
        self.target_tensor = feature_calculator(self.args, self.graph)
        self.target_tensor = torch.FloatTensor(self.target_tensor)
        self.adjacency_opposite = adjacency_opposite_calculator(self.graph)
        self.adjacency_opposite = torch.FloatTensor(self.adjacency_opposite)
        self.model = AttentionWalkLayer(self.args, self.target_tensor.shape)


    def fit(self):
        print("\n\nTraining the model.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        for i in tqdm(range(0,1000)):
            self.optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            self.optimizer.step()
            print(loss.item())
    def save_embedding(self):
        print("\n\nSaving the model.\n")
        pass  


