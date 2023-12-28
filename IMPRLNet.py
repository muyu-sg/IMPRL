import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
import sys

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb


class IMPRLNet_classify(nn.Module):
    def __init__(self, nfeats, n_view, n_classes, args, PHI, block, device):  # phi-> n x n
        super(IMPRLNet_classify, self).__init__()
        self.n_classes = n_classes
        self.n_view = n_view
        self.beta = args.beta  # X.t X X
        self.delta = args.delta  # -delra/L phi Z
        self.block = block
        self.device = device
        self.fusion_type = args.fusion_type
        self.input_type = args.input_type
        self.phi = torch.Tensor(PHI)
        self.theta = nn.Parameter(torch.FloatTensor([args.thre]), requires_grad=True).to(device)
        self.bn_input_01 = nn.BatchNorm1d(self.n_classes, momentum=0.5).to(device)
        self.init_S = []
        for i in range(n_view):
            self.Ss = nn.Linear(nfeats[i], self.n_classes, bias=False).to(device)
            self.init_S.append(self.Ss)
        if self.input_type == 'feature':
            for i in range(n_view):
                exec('self.block{} = DBlock(n_classes, {}, args, self.phi, device)'.format(i, nfeats[i]))
        elif self.input_type == 'similarity':
            for i in range(n_view):
                exec('self.block{} = DBlock(n_classes, n, device)'.format(i, nfeats[i]))
        else:
            sys.exit("Please using a correct input matrix type")

    def self_active_sh(self, x, thershold):
        return F.leaky_relu(x - 1 / thershold) - F.leaky_relu(-1.0 * x - 1 / thershold)

    def forward(self, features, adj):
        Z_init = 0
        for j in range(self.n_view):
            Z_init += self.init_S[j](torch.FloatTensor(features[j] / 1.0).to(self.device))
        output_z = []
        output_z.append(self.self_active_sh(Z_init, self.theta))
        L_norm = torch.norm(output_z[-1].mm(output_z[-1].t()))
        for i in range(0, self.block):
            H_list = []
            Z_temp = 0
            for j in range(0, self.n_view):
                if self.input_type == 'feature':
                    exec('h{} = self.block{}(output_z[-1], adj[{}], features[{}] / 1.0, L_norm, self.device)'.format(j, j, j, j))
                elif self.input_type == 'similarity':
                    exec('h{} = self.block{}(output_z[-1], lap[{}], sim[{}] / 1.0)'.format(j, j, j, j))
                exec('H_list.append(h{})'.format(j))
                Z_temp += H_list[-1]
            Z_temp = Z_temp / self.n_view
            output_z.append(self.self_active_sh(Z_temp, self.theta))
        return output_z[1:]

class DBlock(Module):
    # differentiable network block
    def __init__(self, out_features, nfeats, args, phi, device):
        super(DBlock, self).__init__()
        # self.S_l = []
        # u,w -> c x c
        self.beta = args.beta # X.t X X
        self.theta = args.thre  # 阈值
        self.delta = args.delta
        self.phi = phi.to(device)
        self.U = nn.Linear(out_features, out_features, bias=False).to(device)
        self.W = nn.Linear(out_features, out_features, bias=False).to(device)
        self.S = nn.Linear(nfeats, out_features, bias=False).to(device)

        self.device = device

    def forward(self, input, adj, fea, L, device):
        # self.to(device)
        input1 = self.U(input)  # uz
        input2 = self.S(fea)  # xs
        input3 = adj.mm(self.W(input))
        X = torch.Tensor(fea.to(device) / 1.0).mm(torch.Tensor(fea.to(device) / 1.0).T)
        B = (self.beta / L) * (X.mm(input))
        D = (self.delta / L) * (self.phi.mm(input))
        output = input1 + input2 - input3 + B - D
        return output
