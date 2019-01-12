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

    def forward(self, weighted_target_tensor):
        if self.args.geometric == True:
            print("Geometric")
        else:
            self.attention_probs = torch.nn.functional.softmax(self.attention, dim = 0)


        weighted_target_tensor = weighted_target_tensor * self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_matrix = torch.sum(weighted_target_tensor, dim=0).view(self.shapes[1],self.shapes[2])
        loss_on_target = - weighted_target_matrix * torch.log(torch.nn.functional.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        
        

class AttentionWalkTrainer(object):
    '''
    '''
    def __init__(self, args):
        self.args = args
        self.graph = read_graph(self.args.edge_path)
        self.initialize_model()

    def initialize_model(self):
        self.target_tensor = feature_calculator(self.args, self.graph)
        self.adjacency_opposite = adjacency_opposite_calculator(self.graph)
        self.target_tensor = torch.FloatTensor(self.target_tensor)
        self.model = AttentionWalkLayer(self.args, self.target_tensor.shape)


    def fit(self):
        print("\n\nTraining the model.\n")
        for i in tqdm(range(0,1000)):
            self.model(self.target_tensor)

    def save_embedding(self):
        print("\n\nSaving the model.\n")
        pass  


