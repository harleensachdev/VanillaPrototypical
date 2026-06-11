import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import tqdm
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROTO_WEIGHT, RELATION_WEIGHT, N_WAY, N_SUPPORT, N_QUERY, EPISODES, LEARNING_RATE

from src.dataset import(EpisodicDataLoader)

def train_few_shot(model, dataset, episodes=EPISODES, n_way=N_WAY, n_support=N_SUPPORT, n_query=N_QUERY, relation_weight=RELATION_WEIGHT, proto_weight=PROTO_WEIGHT, **kwargs):
    """
    Train the few-shot ensemble model using pre-created support and query loaders
    
    Args:
        model: EnsembleModel instance
        dataset: Dataset to sample episodes from
        episodes: Number of training episodes
        n_way: Number of classes per episode
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        relation_weight: Weight for relation network loss
        proto_weight: Weight for prototypical network loss
        **kwargs: Additional arguments for loss function creation
    
    Returns:
        List of training losses
    """

    # Create episodic dataloader 
    episodic_loader = EpisodicDataLoader(
        dataset=dataset,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        episodes=episodes
    )
    
    # loss functions
    proto_criterion = nn.NLLLoss()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    pbar = tqdm.tqdm(range(episodes), desc="Training Episodes")

    # Training loop
    train_losses = []
    
    device = next(model.parameters()).device
    
    for episode in pbar:
        model.train()
        
        # Get a batch (episode) from loader
        support_set, query_set = next(iter(episodic_loader))

        # Unpack support + query set
        support_data, support_labels = support_set
        query_data, query_labels = query_set

        # adding channel dims
        if support_data.dim() == 3:
            support_data = support_data.unsqueeze(1)
        if query_data.dim() == 3:
            query_data = query_data.unsqueeze(1)
        
        # Move data to device
        support_data = support_data.to(device)
        support_labels = support_labels.to(device)
        query_data = query_data.to(device)
        query_labels = query_labels.to(device)
        
        # get encoder embeddings
        support_embeddings = model.encoder(support_data)
        query_embeddings = model.encoder(query_data)

        # calculate prototypes (protonet)
        prototypes = torch.zeros(n_way, support_embeddings.shape[1], device=device)
        for i in range(n_way):
            # find indices for each class in support set
            class_indices = torch.where(support_labels == i)[0]
            # calculate prototype
            prototypes[i] = support_embeddings[class_indices].mean(0)
        
        # prototypical loss
        dists = torch.cdist(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)

        # get target label (0 to n_way - 1)
        target_inds = query_labels % n_way

        proto_loss = proto_criterion(log_p_y, target_inds)

        total_loss =proto_loss 

        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # record loss
        train_losses.append(total_loss.item())

        pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

    return train_losses  