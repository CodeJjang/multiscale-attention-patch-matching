import torch.nn as nn

from networks.MultiscaleTransformerEncoder import MultiscaleTransformerEncoder


class PatchMatchingNetwork(nn.Module):

    def __init__(self, output_attention_weights=False):
        super(PatchMatchingNetwork, self).__init__()

        encoder_dim = 128
        self.multiscale_transformer_encoder = MultiscaleTransformerEncoder(encoder_dim, pos_encoding_dim=20,
                                                                           output_attention_weights=output_attention_weights)

        self.output_attention_weights = output_attention_weights

    def forward(self, input1, input2=None, dropout=0.0):

        if input1.nelement() == 0:
            return 0

        res = dict()
        if not self.output_attention_weights:
            res['Emb1'] = self.multiscale_transformer_encoder(input1, dropout=dropout)
            if input2 is not None:
                res['Emb2'] = self.multiscale_transformer_encoder(input2, dropout=dropout)
        else:
            res['Emb1'], res['Emb1Attention'] = self.multiscale_transformer_encoder(input1, dropout=dropout)
            if input2 is not None:
                res['Emb2'], res['Emb2Attention'] = self.multiscale_transformer_encoder(input2, dropout=dropout)
        return res

