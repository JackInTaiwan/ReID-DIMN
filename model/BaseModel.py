import torch

from torch import nn
from .component import Encoder, MappingNetwork, MemoryBank



class BaseModel(nn.Module):
    """
    This is Domain-Invariant-Mapping-Network model.
    """
    def __init__(self, cls_size, params):
        super().__init__()

        feature_dim = params["feature_dim"]
        self.encoder = Encoder(params=params["encoder"])
        self.classifier = nn.Linear(feature_dim, cls_size, bias=False)


    def forward(self, x, cls_):
        """
        Args:
        
        Return:
        
        """
        features = self.encoder(x)
        logits_classifier = self.classifier(features)

        return features, logits_classifier


    def inference(self, x):
        """
        Args:

        Return:

        """
        features = self.encoder.inference(x)
        return features
