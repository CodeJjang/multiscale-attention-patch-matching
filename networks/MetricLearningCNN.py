import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from networks.ConvNet import ConvNet
from layers.L2Norm import L2Norm


class MetricLearningCNN(nn.Module):
    def __init__(self, mode, use_gru, arch_version):
        super(MetricLearningCNN, self).__init__()

        self.mode = mode

        self.netS = ConvNet()
        self.netAS1 = ConvNet()
        self.netAS2 = ConvNet()

        self.fc1 = Linear(128 * 2, 128)
        self.fc2 = Linear(128 * 2, 128)
        self.fc3 = Linear(128 * 2, 4)

        self.fc2 = copy.deepcopy(self.fc1)

        K = 16
        self.fc1A = Linear(128 * 2, K)
        self.fc1B = Linear(K, 128)

        self.fc2A = Linear(128 * 2, K)
        self.fc2B = Linear(K, 128)

        self.use_gru = use_gru
        if use_gru:
            self.gru_layers = 2
            self.arch_version = arch_version
            if arch_version == 1:
                self.gru = nn.GRU(128, 128, num_layers=self.gru_layers, batch_first=True)
            elif arch_version == 2:
                self.gru = nn.GRU(128, 128, num_layers=self.gru_layers, batch_first=True)
                self.fc4 = Linear(128, 1)

    def separate_cnn_params(self, modules):
        all_parameters = modules.parameters()
        params_only_bn = []

        for pname, p in modules.named_parameters():
            if pname.find('bn') >= 0:
                params_only_bn.append(p)

        params_only_bn_id = list(map(id, params_only_bn))
        params_wo_bn = list(filter(lambda p: id(p) not in params_only_bn_id, all_parameters))

        # return params_only_bn, paras_wo_bn
        return params_wo_bn

    def SymmCnnParams(self):
        netSParams = self.separate_cnn_params(self.netS)
        return netSParams

    def AsymmCnnParams(self):
        netAS1Params = self.separate_cnn_params(self.netAS1)
        netAS2Params = self.separate_cnn_params(self.netAS2)

        Params = netAS1Params + netAS2Params

        return Params

    def BaseCnnParams(self):
        netSParams = self.separate_cnn_params(self.netS)
        netAS1Params = self.separate_cnn_params(self.netAS1)
        netAS2Params = self.separate_cnn_params(self.netAS2)

        Params = netSParams + netAS1Params + netAS2Params

        return Params

    def HeadCnnParams(self):
        fc1Params = self.separate_cnn_params(self.fc1)
        fc2Params = self.separate_cnn_params(self.fc2)

        Params = fc1Params + fc2Params

        return Params

    def GetChannelCnn(self, channel, mode):

        if (mode == 'TripletSymmetric') | (mode == 'PairwiseSymmetric'):
            return self.netS

        if (mode == 'TripletAsymmetric') | (mode == 'PairwiseAsymmetric'):
            if channel == 0:
                return self.netAS1

            if channel == 1:
                return self.netAS2

        if (mode == 'Hybrid') | (mode == 'Hybrid1') | (mode == 'Hybrid2'):
            if channel == 0:
                self.mode = 'Hybrid1'
                return self

            if channel == 1:
                self.mode = 'Hybrid2'
                return self

    def freeze_symmetric_cnn(self):
        self.netS.freeze()

    def freeze_asymmetric_cnn(self):
        self.netAS1.freeze()
        self.netAS2.freeze()

    def pairwise_symmetric_gru(self, S1A, S1B):
        output1 = self.netS(S1A, mode='NoFC')  # (384, 128, 29, 29)
        output2 = self.netS(S1B, mode='NoFC')
        batch_size = output1.shape[0]
        feature_size = output1.shape[1]
        sequence_len = output1.shape[2] * output1.shape[3]
        output1 = output1.reshape(batch_size, sequence_len, feature_size)
        output2 = output2.reshape(batch_size, sequence_len, feature_size)
        if self.arch_version == 1:
            _, gru_last_output1 = self.gru(output1)
            _, gru_last_output2 = self.gru(output2)
            # Extract output of second layer
            gru_last_output1 = gru_last_output1.reshape(self.gru_layers, -1, batch_size, feature_size)[1]
            gru_last_output2 = gru_last_output2.reshape(self.gru_layers, -1, batch_size, feature_size)[1]
            output1 = gru_last_output1.squeeze()
            output2 = gru_last_output2.squeeze()
        elif self.arch_version == 2:
            gru_output1, _ = self.gru(output1)  # (384, 128, 841), (2, 384, 841)
            gru_output2, _ = self.gru(output2)
            gru_output1 = F.softmax(gru_output1, dim=2)
            gru_output2 = F.softmax(gru_output2, dim=2)
            output1 = gru_output1 * output1.reshape(*gru_output1.shape)
            output2 = gru_output2 * output2.reshape(*gru_output2.shape)
            output1 = self.fc4(output1).squeeze()
            output2 = self.fc4(output2).squeeze()
        output1 = L2Norm()(output1)  # (384, 128)
        output2 = L2Norm()(output2)
        return output1, output2

    # output CNN
    def forward(self, S1A, S1B=0, mode=-1, Rot1=0, Rot2=0, p=0.0):

        if (S1A.nelement() == 0):
            return 0

        if mode == -1:
            mode = self.mode

        if mode == 'PairwiseSymmetric':
            if self.use_gru:
                output1, output2 = self.pairwise_symmetric_gru(S1A, S1B)
            else:
                # source#1: vis
                output1 = self.netS(S1A)  # (384, 128)
                output2 = self.netS(S1B)
                # summary(self.netS, (1, 64, 64))
            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2
            return Result

        if mode == 'PairwiseSymmetricRot':
            # source#1: vis

            # unnormalized
            output1 = self.netS.net(S1A)
            output2 = self.netS.net(S1B)

            # normalized
            Norm1 = F.normalize(output1, dim=1, p=2)
            Norm2 = F.normalize(output2, dim=1, p=2)

            # rotated unnormalized
            Rot1 = self.netS.net(Rot1)
            Rot2 = self.netS.net(Rot2)

            Result = dict()
            Result['Unnormalized1'] = output1
            Result['Unnormalized2'] = output2

            Result['Emb1'] = Norm1
            Result['Emb2'] = Norm2

            Result['RotUnnormalized1'] = Rot1
            Result['RotUnnormalized2'] = Rot2

            return Result

        if mode == 'PairwiseAsymmetric':
            # source#1: vis
            output1 = self.netAS1(S1A)
            output2 = self.netAS2(S1B)

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2
            return Result

        if (mode == 'Hybrid') | (mode == 'HybridCat'):
            # p: probability of an element to be zeroed.Default: 0.5
            p = 0.0

            Result = dict()

            # source#1: vis
            # channel1
            EmbSym1 = self.netS(S1A, 'Normalized')
            EmbAsym1 = self.netAS1(S1A, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            # if mode == 'HybridCat':
            #    Result['Hybrid1'] = Hybrid1

            Hybrid1 = F.relu(Hybrid1)
            Hybrid1 = Dropout(p)(Hybrid1)  # 20% probabilit
            # Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            # prepare output
            Hybrid1 = self.fc1(Hybrid1)
            # Hybrid1 = self.fc1B(self.fc1A(Hybrid1))

            #

            # Hybrid1 = F.relu((self.fc1A(Hybrid1))
            # Hybrid1 = self.fc1B(Hybrid1)

            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)
            # Hybrid1 = L2Norm()(Hybrid1)

            # channel2
            EmbSym2 = self.netS(S1B, 'Normalized')
            EmbAsym2 = self.netAS2(S1B, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            # if mode == 'HybridCat':
            #   Result['Hybrid2'] = Hybrid2
            #  return Result

            Hybrid2 = F.relu(Hybrid2)
            Hybrid2 = Dropout(p)(Hybrid2)

            # prepare output
            Hybrid2 = self.fc2(Hybrid2)

            # Hybrid2 = F.relu((self.fc2A(Hybrid2))
            # Hybrid2 = self.fc2B(Hybrid2)

            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)
            # Hybrid2 = L2Norm()(Hybrid2)

            Result['Hybrid1'] = Hybrid1
            Result['Hybrid2'] = Hybrid2
            Result['EmbSym1'] = EmbSym1
            Result['EmbSym2'] = EmbSym2
            Result['EmbAsym1'] = EmbAsym1
            Result['EmbAsym2'] = EmbAsym2
            return Result

        if (mode == 'Hybrid1') | (mode == 'HybridCat1'):
            # source#1: vis
            # channel1
            EmbSym1 = self.netS(S1A, 'Normalized')
            EmbAsym1 = self.netAS1(S1A, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            if mode == 'HybridCat':
                return Hybrid1

            Hybrid1 = F.relu(Hybrid1)
            # Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            # prepare output
            Hybrid1 = self.fc1(Hybrid1)
            # Hybrid1 = self.fc1B(self.fc1A(Hybrid1))
            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)
            # Hybrid1 = L2Norm()(Hybrid1)

            return Hybrid1

        if (mode == 'Hybrid2') | (mode == 'HybridCat2'):
            # source#2: IR
            EmbSym2 = self.netS(S1A, 'Normalized')
            EmbAsym2 = self.netAS2(S1A, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            if mode == 'HybridCat2':
                return Hybrid2

            Hybrid2 = F.relu(Hybrid2)
            # Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            # prepare output
            Hybrid2 = self.fc2(Hybrid2)
            # Hybrid2 = self.fc2B(self.fc2A(Hybrid2))
            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)
            # Hybrid2 = L2Norm()(Hybrid2)

            return Hybrid2

        if mode == 'SM':
            z = F.relu(torch.cat((S1A, S1B), 1))
            RotFlag = self.fc3(z)

            return RotFlag
