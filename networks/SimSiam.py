import torch.nn as nn
from torch.nn import functional as F

from layers.PredictionMLP import PredictionMLP
from layers.ProjectionMLP import ProjectionMLP
from networks.losses import ContrastiveLoss


class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        # self.projector = ProjectionMLP(in_dim=backbone.output_dim)
        self.projector = ProjectionMLP(in_dim=8192)

        self.encoder = self.backbone
        self.predictor = PredictionMLP()
        # self.loss = ContrastiveLoss()

    def forward(self, x1, x2):
        enc_output = self.encoder(x1, x2)
        z1, z2 = enc_output['Emb1'], enc_output['Emb2']
        z1, z2 = F.normalize(z1, dim=1, p=2), F.normalize(z2, dim=1, p=2)
        if self.training:
            z1, z2 = self.projector(z1), self.projector(z2)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # loss = self.loss(p1, z2) / 2 + self.loss(p2, z1) / 2
            # return loss
            z1, z2 = p1, p2
        res = {
            'Emb1': z1,
            'Emb2': z2
        }
        return res
