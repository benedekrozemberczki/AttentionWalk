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
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], int(self.args.dimensions/2)))
        self.right_factors = torch.nn.Parameter(torch.Tensor(int(self.args.dimensions/2),self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0],1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.xavier_normal_(self.left_factors)
        torch.nn.init.xavier_normal_(self.right_factors)
        torch.nn.init.xavier_normal_(self.attention)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim = 0)

        weighted_target_tensor = weighted_target_tensor * self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_matrix = torch.sum(weighted_target_tensor, dim=0).view(self.shapes[1],self.shapes[2])
        loss_on_target = - weighted_target_matrix * torch.log(torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_opposite = - adjacency_opposite * torch.log(1-torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_on_matrices = (loss_on_target + loss_opposite).norm(1)
        #loss_on_matrices = torch.mean(torch.abs(loss_on_target + loss_opposite))
        loss_on_regularization = self.args.beta * (self.attention.norm(2)**2)
        loss = loss_on_matrices +  loss_on_regularization
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
        print("\nTraining the model.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            self.optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            self.optimizer.step()
            self.epochs.set_description("Attention Walk (Loss=%g)" % round(loss.item(),4))

    def save_model(self):
        self.save_embedding()
        self.save_attention()

    def save_embedding(self):
        print("\nSaving the model.\n")
        left = self.model.left_factors.detach().numpy()
        right = self.model.right_factors.detach().numpy().T
        indices = np.array([range(len(self.graph))]).reshape(-1,1)
        embedding = np.concatenate([indices, left, right], axis = 1)
        columns = ["id" ] + ["x_" + str(x) for x in range(self.args.dimensions)]
        embedding = pd.DataFrame(embedding, columns = columns)
        embedding.to_csv(self.args.embedding_path, index = None)

    def save_attention(self):
        attention = self.model.attention_probs.detach().numpy()
        indices = np.array([range(self.args.window_size)]).reshape(-1,1)
        attention = np.concatenate([indices, attention], axis = 1)
        attention = pd.DataFrame(attention, columns = ["Order","Weight"])
        attention.to_csv(self.args.attention_path, index = None)
