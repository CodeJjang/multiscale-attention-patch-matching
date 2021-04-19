import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from networks.BackboneCNN import BackboneCNN
from networks.positional_encodings import prepare_2d_pos_encodings
from networks.transformer import TransformerEncoderLayer, TransformerEncoder


class MultiscaleTransformerEncoder(nn.Module):

    def __init__(self, dropout, encoder_dim=128, pos_encoding_dim=20, output_attention_weights=False):
        super(MultiscaleTransformerEncoder, self).__init__()

        self.backbone_cnn = BackboneCNN(output_feat_map=True, dropout=dropout)

        self.query = nn.Parameter(torch.randn(1, encoder_dim))
        self.query_pos_encoding = nn.Parameter(torch.randn(1, encoder_dim))

        self.pos_encoding_x = nn.Parameter(torch.randn(pos_encoding_dim, int(encoder_dim / 2)))
        self.pos_encoding_y = nn.Parameter(torch.randn(pos_encoding_dim, int(encoder_dim / 2)))

        # spp levels; first feature map will be passed in a residual connection to the output
        self.spp_levels = [8, 8, 4, 2, 1]
        # self.spp_levels = [8, 4, 2, 1]

        encoder_layers = 2
        encoder_heads = 2

        self.encoder_layer = TransformerEncoderLayer(d_model=encoder_dim, nhead=encoder_heads,
                                                     dim_feedforward=int(encoder_dim),
                                                     dropout=0.1, activation="relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=encoder_layers)

        # self.SPP_FC = nn.Linear(8576, encoder_dim) # for [8,4,2,1] SPP
        self.SPP_FC = nn.Linear(8704, encoder_dim)  # for [8,8,4,2,1] SPP
        self.output_attention_weights = output_attention_weights

    def encoder_spp(self, previous_conv, num_sample, previous_conv_size):
        attention_weights = []
        for i in range(len(self.spp_levels)):

            # Pooling support
            h_wid = int(math.ceil(previous_conv_size[0] / self.spp_levels[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / self.spp_levels[i]))

            # Padding to retain orthogonal dimensions
            h_pad = int((h_wid * self.spp_levels[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.spp_levels[i] - previous_conv_size[1] + 1) / 2)

            # apply pooling
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            y = maxpool(previous_conv)

            if i == 0:
                spp = y.reshape(num_sample, -1)
            else:
                pos_encoding_2d = prepare_2d_pos_encodings(self.pos_encoding_x,
                                                           self.pos_encoding_y,
                                                           y.shape[2], y.shape[3])

                pos_encoding = pos_encoding_2d.permute(2, 0, 1)
                pos_encoding = pos_encoding[:, 0:y.shape[2], 0:y.shape[3]]
                pos_encoding = pos_encoding.reshape(
                    (pos_encoding.shape[0], pos_encoding.shape[1] * pos_encoding.shape[2]))
                pos_encoding = pos_encoding.permute(1, 0).unsqueeze(1)
                pos_encoding = torch.cat((self.query_pos_encoding.unsqueeze(0), pos_encoding), 0)

                seq = y.reshape((y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
                seq = seq.permute(2, 0, 1)

                query = self.query.repeat(1, seq.shape[1], 1)
                seq = torch.cat((query, seq), 0)

                enc_output = self.encoder(src=seq, pos=pos_encoding)

                seq = enc_output[0,]
                if self.output_attention_weights and i == 1:
                    attention_weights = enc_output[1:].transpose(1, 0)

                spp = torch.cat((spp, seq.reshape(num_sample, -1)), 1)

        if self.output_attention_weights:
            return spp, attention_weights
        return spp

    def forward_one(self, x):

        activ_map = self.backbone_cnn(x)

        spp_result = self.encoder_spp(activ_map, x.size(0),
                                      [int(activ_map.size(2)), int(activ_map.size(3))])
        if self.output_attention_weights:
            spp_activations = spp_result[0]
            attention_weights = spp_result[1]
        else:
            spp_activations = spp_result

        res = self.SPP_FC(spp_activations)
        res = F.normalize(res, dim=1, p=2)

        if self.output_attention_weights:
            return res, attention_weights
        return res

    def forward(self, x1, x2):
        res = dict()
        if not self.output_attention_weights:
            res['Emb1'] = self.forward_one(x1)
            res['Emb2'] = self.forward_one(x2)
        else:
            res['Emb1'], res['Emb1Attention'] = self.forward_one(x1)
            res['Emb2'], res['Emb2Attention'] = self.forward_one(x2)
        return res
