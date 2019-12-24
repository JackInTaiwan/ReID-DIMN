import torch

from torch import nn
from .component import Encoder, MappingNetwork, MemoryBank



class DIMNModel(nn.Module):
    """
    This is Domain-Invariant-Mapping-Network model.
    """
    def __init__(self, cls_size, params):
        super().__init__()

        feature_dim = params["feature_dim"]
        self.encoder = Encoder(params=params["encoder"])
        self.mapping_network = MappingNetwork(in_dim=feature_dim)
        self.memory_bank = MemoryBank(cls_size=cls_size, feature_dim=feature_dim, params=params["memory_bank"])
        self.classifier = nn.Linear(feature_dim, cls_size, bias=False)


    def forward(self, x_probe, x_gallery, cls_):
        """
        Args:
        
        Return:
        
        """
        x = torch.cat((x_probe, x_gallery), dim=0)
        features = self.encoder(x)
        features_probe, features_gallery = features[:x_probe.size(0)], features[x_probe.size(0):]
        pred_cls_weights = self.mapping_network(features_gallery)
        selected_cls_weights, tmp_memory = self.memory_bank(pred_cls_weights, cls_)
        logits_classifier = self.classifier(features_probe)
        logits_mapping_network = torch.bmm(tmp_memory, features_probe.unsqueeze(2)).squeeze(2) # (B, C, 1)

        return features_probe, logits_classifier, logits_mapping_network, pred_cls_weights, selected_cls_weights, tmp_memory


    def inference_gallery(self, x, cls_):
        """
        Args:

        Return:

        """
        features = self.encoder.inference(x)
        pred_cls_weights = self.mapping_network.inference(features)
        self.memory_bank.inference(pred_cls_weights, cls_)


    def inference_probe(self, x):
        """
        Args:

        Return:

        """
        features = self.encoder.inference(x)
        # FIXME: ugly codes
        with torch.no_grad():
            logits_classifier = self.classifier(features)
        
        pred_logits = torch.bmm(self.memory_bank.memory.repeat(x.size(0), 1, 1), features.unsqueeze(2)).squeeze(2) # (B, C, 1)

        return pred_logits 
