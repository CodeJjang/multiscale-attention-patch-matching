import torch.nn as nn

from networks.MultiscaleTransformerEncoder import MultiscaleTransformerEncoder


class PatchMatchingNetwork(nn.Module):

    def __init__(self, dropout, output_attention_weights=False):
        super(PatchMatchingNetwork, self).__init__()

        self.multiscale_transformer_encoder = MultiscaleTransformerEncoder(dropout, pos_encoding_dim=20,
                                                                           output_attention_weights=output_attention_weights)

        self.output_attention_weights = output_attention_weights

    def forward(self, input1, input2):

        if input1.nelement() == 0:
            return 0

        res = dict()
        if not self.output_attention_weights:
            res['Emb1'] = self.multiscale_transformer_encoder(input1)
            res['Emb2'] = self.multiscale_transformer_encoder(input2)
        else:
            res['Emb1'], res['Emb1Attention'] = self.multiscale_transformer_encoder(input1)
            res['Emb2'], res['Emb2Attention'] = self.multiscale_transformer_encoder(input2)
        return res
