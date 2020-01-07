import torch

from torch import nn



class RegularizationLoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()

        self.reduction = reduction
    

    def forward(self, pred_cls_weights, memory_cls_weights):
        loss = (pred_cls_weights - memory_cls_weights).norm(dim=1) ** 2
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        
        return loss



class TripletLoss(nn.Module):
    def __init__(self, delta, reduction="sum"):
        super().__init__()

        self.reduction = reduction
        self.delta = delta
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, logits_mapping_network, cls_list):
        device = logits_mapping_network.device
        logits = logits_mapping_network  # (B, C)
        prob = self.softmax(logits)    # (B, C)

        mask = torch.zeros((prob.size(0), prob.size(1)), dtype=torch.bool).to(device)
        for i, cls_index in enumerate(cls_list):
            mask[i][cls_index] = 1
        prob_pos = prob.masked_select(mask)
        prob_neg = prob.masked_fill(mask, value=0)
        max_prob_neg, _ = torch.max(prob_neg, dim=1)
        loss, _ = torch.max(torch.stack([torch.zeros_like(max_prob_neg).to(device), self.delta + max_prob_neg - prob_pos]), dim=0)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
