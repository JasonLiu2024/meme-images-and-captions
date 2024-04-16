import torch
import torch.nn as nn
import help
from typing import Union, Any
from torch.utils.data import dataloader

arch : dict[str, int] = {
    'image_embedding': 1
}

class Siamese(nn.Module):
    """
    Twin inputs, 
    L1 distance combines embeddings,
    Binary Cross Entropy loss"""
    def __init__(self, feature_dim : int=100,
            image_embedding_dim: int=2048, caption_embedding_dim: int=384,
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
        self.ending = nn.Sequential(
            torch.nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.projection_layer.apply(self.init_weights)
        self.shared.apply(self.init_weights)
        # self.ending.apply(self.init_weights)
        self.projection_layer.to(self.device)
        self.shared.to(self.device)
        self.ending.to(self.device)

    def forward(self, image_embedding: int, caption_embedding: int):
        image_embedding_in_caption_embedding_space = self.projection_layer(image_embedding)
        image_feature = self.shared(image_embedding_in_caption_embedding_space)
        caption_feature = self.shared(caption_embedding)
        """TODO: different ways to combine features"""
        l1_distance = torch.abs(image_feature - caption_feature)
        return self.ending(l1_distance)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
    
def train(model : nn.Module, dataloader_train: dataloader.DataLoader,
        hyperparameters: dict[str, Any], dataloader_validate: Union[dataloader.DataLoader, None]=None):
    print(f"TRAINing: START")
    model.train()
    optimizer: torch.optim.Optimizer = hyperparameters['optimizer_type'](
        params=model.parameters(), lr=hyperparameters['learning_rate'])
    losses = []
    loss_function:nn.Module = hyperparameters['loss_function']()
    for i_step, batch in enumerate(dataloader_train):
        # print(f"TRAIN(): {i_step}")
        image_embedding_batch, caption_embedding_batch, labels_batch = batch
        optimizer.zero_grad()
        prediction_batch = model(image_embedding_batch, caption_embedding_batch)
        loss: torch.Tensor = loss_function(prediction_batch, labels_batch)
        loss.backward()
        optimizer.step()
        loss.detach()
        losses.append(loss.item())
    return losses
    

from torcheval.metrics.functional import binary_auprc, binary_auroc

def test(model: nn.Module, dataloader_test: dataloader.DataLoader, hyperparameters: dict[str, Any]):
    model.eval()
    losses, auprcs, aurocs = [], [], []
    correct = 0
    loss_function: nn.Module = hyperparameters['loss_function']()
    with torch.no_grad():
        for i_step, batch in enumerate(dataloader_test):
            image_embedding_batch, caption_embedding_batch, labels_batch = batch
            prediction_batch = model(image_embedding_batch, caption_embedding_batch)
            loss : torch.Tensor = loss_function(prediction_batch, labels_batch)
            loss.detach()
            losses.append(loss.item())
            prediction_batch_bools = torch.where(prediction_batch > 0.5, 1, 0)  # get the index of the max log-probability
            correct += prediction_batch_bools.eq(labels_batch.view_as(prediction_batch_bools)).sum().item()

            prediction_batch_bools = torch.where(prediction_batch > 0.5, 1, 0)  # get the index of the max log-probability
            correct += prediction_batch_bools.eq(labels_batch.view_as(prediction_batch_bools)).sum().item()
            prediction_batch.detach()
            aurocs.append(binary_auroc(input=torch.squeeze(prediction_batch), target=torch.squeeze(labels_batch)).item())
            auprcs.append(binary_auprc(input=torch.squeeze(prediction_batch), target=torch.squeeze(labels_batch)).item())
        accuracy = 100. * correct / len(dataloader_test.dataset)
    return losses, accuracy, aurocs, auprcs
