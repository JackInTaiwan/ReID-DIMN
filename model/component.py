import torch

from torch import nn
from torchvision.models import MobileNetV2



class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.features = MobileNetV2(
            num_classes=0,
            width_mult=params["features.width_mult"]
        ).features
    

    def forward(self, x):
        """
        """
        features = self.features(x)     # (batch, output_channel=1792, h=7, w=7)
        features = features.mean([2, 3])    # (batch, output_channel=1792)
        
        return features
    

    def inference(self, x):
        with torch.no_grad():
            features = self.features(x)     # (batch, output_channel=1792, h=7, w=7)
            features = features.mean([2, 3])    # (batch, output_channel=1792)
        
        return features



class MappingNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.hyper_network = nn.Linear(in_dim, in_dim, bias=False)


    def forward(self, x):
        weights = self.hyper_network(x)
        
        return weights


    def inference(self, x):
        with torch.no_grad():
            weights = self.hyper_network(x)
        
        return weights



class MemoryBank(nn.Module):
    def __init__(self, cls_size, feature_dim, params):
        super().__init__()

        self.alpha = params["alpha"]
        self.memory = nn.Parameter(torch.zeros(cls_size, feature_dim), requires_grad=False)
        self.tmp_memory = None


    def clean_memory_bank(self):
        self.memory.data.zero_()
        self.tmp_memory = None
    

    def update_tmp_memory(self, x, cls_list):
        self.tmp_memory = self.memory.data.clone()
        selected_cls_weights = []
        for (weight, index) in zip(x, cls_list):
            selected_cls_weights.append(self.memory[index].clone())
            self.tmp_memory[index] = weight
        
        selected_cls_weights = torch.stack(selected_cls_weights)
        
        return selected_cls_weights
                

    def update_memory(self):
        self.memory.data = self.memory.data * (1 - self.alpha) + self.tmp_memory.detach().clone() * self.alpha 
        # column-wise l2 normalization
        self.memory.data = self.memory.data / ((self.memory.data.norm(dim=1).unsqueeze(1) + 1e-7))
        

    def forward(self, x, cls_list):
        # update tmp memory
        selected_cls_weights = self.update_tmp_memory(x, cls_list)
        tmp_memory = self.tmp_memory.repeat(x.size(0), 1, 1)

        # update memory
        self.update_memory()

        return selected_cls_weights, tmp_memory
    

    def inference(self, x, cls_list):
        with torch.no_grad():
            # update tmp memory
            _ = self.update_tmp_memory(x, cls_list)
            # update memory
            self.update_memory()



if __name__ == "__main__":
    encoder = Encoder({})
    x = torch.randn((5, 3, 224, 224))
    features = encoder(x)
    print(features.size())
    memory_bank = MemoryBank(100, 1792, {"alpha": 0.4})
    # x = torch.randn((5, 1792))
    cls_list = [3, 4, 9, 0, 50]
    features = memory_bank(features, cls_list)
    print(features)
    print(features.size())