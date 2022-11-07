# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, args, base_encoder, dim=256, pred_dim=256):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        self.encoder = base_encoder(args)

    def forward(self, x0, x1):
        """
        Input:
            x0: first views of images
            x1: second views of images
        Output:
            p0, p1, z0, z1: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        if self.training:
            # compute features for one view
            z0, p0, k0 = self.encoder(x0) # NxC, NxC, NxK
            z1, p1, k1 = self.encoder(x1) # NxC, NxC, NxK

            return p0, p1, z0, z1, k0, k1
        else:
            z0, _, k0 = self.encoder(x0)
            
            return z0, k0