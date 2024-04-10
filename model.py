import torch
import torch.nn as nn
import help
from typing import Union
from torch.utils.data import dataloader

arch : dict[str, int] = {
    'image_embedding': 1
}

class Siamese(nn.Module):
    """
    Twin inputs, 
    L1 distance combines embeddings,
    Binary Cross Entropy loss"""
    def __init__(self, image_embedding_dim : int, caption_embedding_dim : int, feature_dim : int, 
            layers : dict[str, int],
            device : Union[str, None]=None):
        super().__init__()
        self.device = device if device != None else help.get_device()
        self.projection_layer = nn.Linear(in_features=image_embedding_dim, out_features=caption_embedding_dim)
        hidden_dim_1 = 100
        self.shared = nn.Sequential(
            torch.nn.Linear(caption_embedding_dim, hidden_dim_1),
            nn.ReLU(),
            torch.nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
            torch.nn.Linear(hidden_dim_1, feature_dim)
        )
        self.sigmoid = nn.Sigmoid()
        self.projection_layer.apply(self.init_weights)
        self.shared.apply(self.init_weights)

    def forward(self, image_embedding : int, caption_embedding : int):
        image_embedding_in_caption_embedding_space = self.projection_layer(image_embedding)
        image_feature = self.shared(image_embedding_in_caption_embedding_space)
        caption_feature = self.shared(caption_embedding)
        """TODO: different ways to combine features"""
        combined = torch.cdist(x1=image_feature, x2=caption_feature, p=1) # L1 distance
        return self.sigmoid(combined)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
def train(model : nn.Module, dataloader_train : dataloader.DataLoader, dataloader_validate : dataloader.DataLoader,
        optimizer: torch.optim.Optimizer, learning_rate: float):
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    losses = []
    loss_function = nn.BCELoss()
    for i_step, batch in enumerate(dataloader_train):
        image_embedding_batch, caption_embedding_batch, labels_batch = batch
        optimizer.zero_grad()
        prediction_batch = model(image_embedding_batch, caption_embedding_batch)
        loss : torch.Tensor = loss_function(prediction_batch, labels_batch)
        loss.backward()
        optimizer.step()
        loss.detach()
        losses.append(loss.item())
    return losses
    
def test(self, model : nn.Module, dataloader_test : dataloader.DataLoader):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for i_step, batch in enumerate(dataloader_test):
            image_embedding_batch, caption_embedding_batch, labels_batch = batch
            prediction_batch = self.forward(image_embedding_batch, caption_embedding_batch)
            loss : torch.Tensor = self.loss_function(prediction_batch, labels_batch)
            loss.detach()
            losses.append(loss.item())
            prediction_batch_bools = torch.where(prediction_batch > 0.5, 1, 0)  # get the index of the max log-probability
            correct += prediction_batch_bools.eq(labels_batch.view_as(prediction_batch_bools)).sum().item()
        accuracy = 100. * correct / len(dataloader_test)
    return losses, accuracy
