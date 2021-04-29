import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

import numpy as np

torch.set_printoptions(precision=2, threshold=float('inf'))


class AGCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, model='agcn', gcn_layer=2, dropout=0.0, relu=0, filt_percent=0.7):
        super(AGCNBlock, self).__init__()
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.sort = 'sample'
        self.model = model
        self.gcns = nn.ModuleList()
        self.gcns.append(GCNBlock(input_dim, hidden_dim, 0, 0, 0, 0, 'relu'))

        for i in range(gcn_layer - 1):      # 2 = 3 - 1
            self.gcns.append(GCNBlock(hidden_dim, hidden_dim, 0, 0, 0, 0, 'relu'))

        self.w_a = nn.Parameter(torch.zeros(1, hidden_dim, 1))      # [1, 64, 1]
        self.w_b = nn.Parameter(torch.zeros(1, hidden_dim, 1))      # [1, 64, 1]
        torch.nn.init.normal_(self.w_a)
        torch.nn.init.uniform_(self.w_b, -1, 1)

        self.pass_dim = hidden_dim          # 64

        self.pool = self.mean_pool

        self.softmax = 'neibor'
        if self.softmax == 'gcn':
            self.att_gcn = GCNBlock(2, 1, 0, 0, 0, 'relu')
        self.khop = 1
        self.adj_norm = 'none'

        self.filt_percent = filt_percent      # 0.5
        self.eps = 1e-10                   # 1e-10

        self.tau_config = 1.0           # 1.0
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.lamda1 = nn.Parameter(torch.tensor(1.0))
        self.lamda2 = nn.Parameter(torch.tensor(1.0))

        self.att_norm = 0

        self.dnorm = 1
        self.dnorm_coe = 1.0

        self.att_out = 0
        self.single_att = 0

    def forward(self, X, adj, mask, is_print=False):         # [128, 6, 256], [128, 6, 6], [128, 6]
        '''
    input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
        adj: adj matrix, [batch,node_num,node_num], dtype=float
        mask: mask for nodes, [batch,node_num]
    outputs:
        out:unormalized classification prob, [batch,hidden_dim]
        H: batch of node hidden features, [batch,node_num,pass_dim]
        new_adj: pooled new adj matrix, [batch, k_max, k_max]
        new_mask: [batch, k_max]
        '''
        hidden = X

        is_print1 = is_print2 = is_print
        if adj.shape[-1] > 100:
            is_print1 = False

        for gcn in self.gcns:       # 3
            hidden = gcn(hidden, adj, mask)     # [128, 6, 256]

        hidden = mask.unsqueeze(2) * hidden     # [128, 6, 256] <- [128, 6, 1] * [128, 6, 256]

        if self.model == 'unet':
            att = torch.matmul(hidden, self.w_a).squeeze()
            att = att / torch.sqrt((self.w_a.squeeze(2) ** 2).sum(dim=1, keepdim=True))
        elif self.model == 'agcn':
            if self.softmax == 'global' or self.softmax == 'mix':
                att_a = torch.matmul(hidden, self.w_a).squeeze() + (mask - 1) * 1e10
                att_a_1 = att_a = torch.nn.functional.softmax(att_a, dim=1)
                if self.dnorm:
                    scale = mask.sum(dim=1, keepdim=True) / self.dnorm_coe
                    att_a = scale * att_a
            if self.softmax == 'neibor' or self.softmax == 'mix':
                att_b = torch.matmul(hidden, self.w_b).squeeze() + (mask - 1) * 1e10        # [128, 6] <- [128, 6, 256] * [1, 256, 1] + ([128, 256] - 1) * 1e10
                att_b_max, _ = att_b.max(dim=1, keepdim=True)       # [128, 1], [128, 1]
                if self.tau_config != -2:
                    att_b = torch.exp((att_b - att_b_max) * torch.abs(self.tau))            # [128, 6]
                else:
                    att_b = torch.exp((att_b - att_b_max) * torch.abs(self.tau_fc(self.pool(hidden, mask))))
                denom = att_b.unsqueeze(2)                  # [128, 6, 1]
                for _ in range(self.khop):                  # 1
                    denom = torch.matmul(adj, denom)        # [128, 6, 1] <- [128, 6, 6] * [128, 6, 1]
                denom = denom.squeeze() + self.eps          # [128, 6]
                att_b = (att_b * torch.diagonal(adj, 0, 1, 2)) / denom      # [128, 6] <- ([128, 6] * [128, 6]) / [128, 6]
                if self.dnorm:
                    if self.adj_norm == 'diag':
                        diag_scale = mask / (torch.diagonal(adj, 0, 1, 2) + self.eps)
                    elif self.adj_norm == 'none':
                        diag_scale = adj.sum(dim=1)         # [128, 6]
                    att_b = att_b * diag_scale              # [128, 6] <- [128, 6] * [128, 6]
                att_b = att_b * mask         # [128, 6] <- [128, 6] * [128, 6]

            if self.softmax == 'global':
                att = att_a
            elif self.softmax == 'neibor' or self.softmax == 'hardnei':
                att = att_b         # [128, 6], min=0, max=1
            elif self.softmax == 'mix':
                att = att_a * torch.abs(self.lamda1) + att_b * torch.abs(self.lamda2)

        Z = hidden          # [128, 6, 256]

        if self.model == 'unet':
            Z = torch.tanh(att.unsqueeze(2)) * Z
        elif self.model == 'agcn':
            if self.single_att:
                Z = Z
            else:
                Z = att.unsqueeze(2) * Z        # [128, 6, 256] <- [128, 6, 1] * [128, 6, 256]

        k_max = int(math.ceil(self.filt_percent * adj.shape[-1]))       # 10 <- 0.7 * 14
        if self.model == 'diffpool':
            k_max = min(k_max, self.diffpool_k)

        k_list = [int(math.ceil(self.filt_percent * x)) for x in mask.sum(dim=1).tolist()]     # [16, 13, 12, 13, ...]

        if self.model != 'diffpool':
            if self.sort == 'sample':
                att_samp = att * mask           # [20, 79] <- [20, 79] * [20, 79]
                att_samp = (att_samp / att_samp.sum(1, keepdim=True)).detach().cpu().numpy()   # [128, 6]
                top_index = []
                for i in range(att.size(0)):
                    top_index.append(torch.LongTensor(np.random.choice(att_samp.shape[1], k_max, p=att_samp[i])))
                top_index = torch.stack(top_index)       # [20, 40]
            elif self.sort == 'random_sample':
                top_index = torch.LongTensor(att.size(0), k_max) * 0
                for i in range(att.size(0)):
                    top_index[i, 0:k_list[i]] = torch.randperm(int(mask[i].sum().item()))[0:k_list[i]]
            else:  # sort
                _, top_index = torch.topk(att, k_max, dim=1)

        new_mask = X.new_zeros(X.shape[0], k_max)           # [128, 5]

        visualize_tools = None

        if self.model == 'unet':
            for i, k in enumerate(k_list):
                for j in range(int(k), k_max):
                    top_index[i][j] = adj.shape[-1] - 1
                    new_mask[i][j] = -1.
            new_mask = new_mask + 1
            top_index, _ = torch.sort(top_index, dim=1)
            assign_m = X.new_zeros(X.shape[0], k_max, adj.shape[-1])
            for i, x in enumerate(top_index):
                assign_m[i] = torch.index_select(adj[i], 0, x)
            new_adj = X.new_zeros(X.shape[0], k_max, k_max)
            H = Z.new_zeros(Z.shape[0], k_max, Z.shape[-1])
            for i, x in enumerate(top_index):
                new_adj[i] = torch.index_select(assign_m[i], 1, x)
                H[i] = torch.index_select(Z[i], 0, x)

        elif self.model == 'agcn':
            assign_m = X.new_zeros(X.shape[0], k_max, adj.shape[-1])        # [128, 5, 6]
            for i, k in enumerate(k_list):      # len(k_list)=128
                for j in range(int(k)):
                    assign_m[i][j] = adj[i][top_index[i][j]]    # [6,]
                    new_mask[i][j] = 1.         # [128, 5]
            assign_m = assign_m / (assign_m.sum(dim=1, keepdim=True) + self.eps)       # [128, 5, 6]

            H = torch.matmul(assign_m, Z)       # [128, 5, 256] <- [128, 5, 6] * [128, 6, 256]

            new_adj = torch.matmul(torch.matmul(assign_m, adj), torch.transpose(assign_m, 1, 2))        # [128, 5, 5] <- [128, 5, 6] * [128, 6, 6] * [128, 6, 5]

        elif self.model == 'diffpool':
            hidden1 = X
            for gcn in self.pool_gcns:
                hidden1 = gcn(hidden1, adj, mask)
            assign_m = X.new_ones(X.shape[0], X.shape[1], k_max) * (-100000000.)
            for i, x in enumerate(hidden1):
                k = min(k_list[i], k_max)
                assign_m[i, :, 0:k] = hidden1[i, :, 0:k]
                for j in range(int(k)):
                    new_mask[i][j] = 1.

            assign_m = torch.nn.functional.softmax(assign_m, dim=2) * mask.unsqueeze(2)
            assign_m_t = torch.transpose(assign_m, 1, 2)
            new_adj = torch.matmul(torch.matmul(assign_m_t, adj), assign_m)
            H = torch.matmul(assign_m_t, Z)

        if self.att_out and self.model == 'agcn':
            if self.softmax == 'global':
                out = self.pool(att_a_1.unsqueeze(2) * hidden, mask)
            elif self.softmax == 'neibor':
                att_b_sum = att_b.sum(dim=1, keepdim=True)
                out = self.pool((att_b / (att_b_sum + self.eps)).unsqueeze(2) * hidden, mask)
        else:
            out = self.pool(hidden, mask)           # [128, 256]

        if self.adj_norm == 'tanh' or self.adj_norm == 'mix':
            new_adj = torch.tanh(new_adj)
        elif self.adj_norm == 'diag' or self.adj_norm == 'mix':
            diag_elem = torch.pow(new_adj.sum(dim=2) + self.eps, -0.5)
            diag = new_adj.new_zeros(new_adj.shape)
            for i, x in enumerate(diag_elem):
                diag[i] = torch.diagflat(x)
            new_adj = torch.matmul(torch.matmul(diag, new_adj), diag)

        visualize_tools = []
        '''
        if (not self.training) and is_print1:

            print('**********************************')
            print('node_feat:',X.type(),X.shape)
            print(X)

            if self.model!='diffpool':
                print('**********************************')
                print('att:',att.type(),att.shape)
                print(att)

                print('**********************************')
                print('top_index:',top_index.type(),top_index.shape)
                print(top_index)



            print('**********************************')
            print('adj:',adj.type(),adj.shape)
            print(adj)

            print('**********************************')
            print('assign_m:',assign_m.type(),assign_m.shape)
            print(assign_m)

            print('**********************************')
            print('new_adj:',new_adj.type(),new_adj.shape)
            print(new_adj)

            print('**********************************')
            print('new_mask:',new_mask.type(),new_mask.shape)
            print(new_mask)
        '''

        if (not self.training) and is_print2:
            if self.model != 'diffpool':
                visualize_tools.append(att[0])
                visualize_tools.append(top_index[0])
            visualize_tools.append(new_adj[0])
            visualize_tools.append(new_mask.sum())

        return out, H, new_adj, new_mask, visualize_tools   # [128, 256], [128, 5, 256], [128, 5, 5], [128, 5], []

    def mean_pool(self, x, mask):       # [20, 40, 64], [20, 40]
        return x.sum(dim=1) / (self.eps + mask.sum(dim=1, keepdim=True))

    def sum_pool(self, x, mask):
        return x.sum(dim=1)

    @staticmethod
    def max_pool(x, mask):
        # output: [batch,x.shape[2]]
        m = (mask - 1) * 1e10
        r, _ = (x + m.unsqueeze(2)).max(dim=1)
        return r


# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0, add_self=0, normalize_embedding=0,
                 dropout=0.0, relu=0, bias=True):
        super(GCNBlock, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim      # 37
        self.output_dim = output_dim    # 64
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())     # [256, 256]
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())        # [256,]
        else:
            self.bias = None

    def forward(self, x, adj, mask):        # [128, 6, 256], [128, 6, 6], [128, 6]
        y = torch.matmul(adj, x)            # [128, 6, 256] <- [128, 6, 6] * [128, 6, 256]
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)    # [128, 6, 256]
        if self.bias is not None:
            y = y + self.bias               # [128, 6, 256]
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index), y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index = []
            ssum = 0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i + 1]] = y[i, 0:index[i]]
            bn_tensor_bf = self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i, 0:index[i]] = bn_tensor_bf[start_index[i]:start_index[i + 1]]
            y = bn_tensor_af
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu == 'relu':
            y = torch.nn.functional.relu(y)    # [128, 6, 256]
        elif self.relu == 'lrelu':
            y = torch.nn.functional.leaky_relu(y, 0.1)
        return y


# experimental function, untested
class masked_batchnorm(nn.Module):
    def __init__(self, feat_dim, epsilon=1e-10):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(feat_dim))
        self.beta = nn.Parameter(torch.zeros(feat_dim))
        self.eps = epsilon

    def forward(self, x, mask):
        '''
        x: node feat, [batch,node_num,feat_dim]
        mask: [batch,node_num]
        '''
        mask1 = mask.unsqueeze(2)
        mask_sum = mask.sum()
        mean = x.sum(dim=(0, 1), keepdim=True) / (self.eps + mask_sum)
        temp = (x - mean) ** 2
        temp = temp * mask1
        var = temp.sum(dim=(0, 1), keepdim=True) / (self.eps + mask_sum)
        rstd = torch.rsqrt(var + self.eps)
        x = (x - mean) * rstd
        return ((x * self.alpha) + self.beta) * mask1








class Classifier(nn.Module):
    def __init__(self, model='agcn', input_dim=2048, hidden_dim=2048, filt_percent=0.7):
        super(Classifier, self).__init__()

        self.model = model
        self.eps = 1e-10
        self.pool = self.mean_pool
        self.num_layers = 4

        if self.model == 'gcn':
            self.gcns = nn.ModuleList()
            x_size = input_dim
            for _ in range(self.num_layers):
                self.gcns.append(GCNBlock(x_size, hidden_dim, 0, 0, 0, 0, 'relu'))
                x_size = hidden_dim

        else:
            self.margin = 0.05
            self.agcn_res = 0
            self.num_layers = 2
            gcn_layer_list = [3] + [1] * (self.num_layers - 1)

            self.agcns = nn.ModuleList()
            x_size = input_dim     # 37

            for i in range(self.num_layers):        # 4
                self.agcns.append(AGCNBlock(x_size, hidden_dim, 'agcn', gcn_layer_list[i], 0, 'relu', filt_percent))
                x_size = self.agcns[-1].pass_dim

    def forward(self, node_feat, mask_node, is_print=False):   # [64, 14, 2048], [64, 14, 14]
        B, N, C = node_feat.shape

        # # # Feature Similarity Graph
        # adj = torch.stack([torch.eye(N) for _ in range(B)]).cuda()    # [64, 14, 14]

        node_feat = F.normalize(node_feat, dim=-1)  # [128, 6, 256]
        adj = node_feat.matmul(node_feat.permute(0, 2, 1))  # [20, 6, 6]
        adj = 0.5 * torch.tanh(adj) + 0.5  # sigmoid kernel    min=0.5, max=1
        adj_ = torch.sum(adj, dim=1)  # [20, 6]
        adj_[adj_ == 0] = 1e-10
        d_inv_sqrt = torch.stack([torch.diag(item) for item in torch.pow(adj_, -0.5)])  # [128, 6, 6]
        adj_hat = d_inv_sqrt.matmul(adj).matmul(d_inv_sqrt)  # [128, 6, 6]
        adj = adj_hat.view(B, N, N)  # [128, 6, 6]

        if self.model == 'gcn':
            return self.gcn_forward(node_feat, adj, mask_node)
        else:
            return self.agcn_forward(node_feat, adj, mask_node, is_print=is_print)

    def mean_pool(self, x, mask):
        return x.sum(dim=1) / (self.eps + mask.sum(dim=1, keepdim=True))

    @staticmethod
    def max_pool(x, mask):
        # output: [batch,x.shape[2]]
        m = (mask - 1) * 1e10
        r, _ = (x + m.unsqueeze(2)).max(dim=1)
        return r

    def gcn_forward(self, node_feat, adj, mask):
        X = node_feat
        vis = []
        for i in range(self.num_layers):
            X = self.gcns[i](X, adj, mask)
            if not self.training:
                vis.append(X.cpu())
        embed = self.pool(X, mask)
        if not self.training:
            vis.append(mask.cpu())
            vis.append(embed.cpu())
            vis = vis[::-1]
        return embed, vis

    def agcn_forward(self, node_feat, adj, mask, is_print=False):        # [128, 14, 2048], [128, 14, 14], [128, 14]
        X = node_feat
        visualize_tools = []
        embeds = []
        Xs = []

        for i in range(self.num_layers):    # 2
            embed, X, adj, mask, visualize_tool = self.agcns[i](X, adj, mask, is_print=is_print)    # [128, 256], [128, 5, 256], [128, 5, 5], [128, 5], []
            embeds.append(embed)
            Xs.append(X)

            # if not self.training:
            #     visualize_tools.append([visualize_tool.cpu(), embed.cpu(), X.cpu(), mask.cpu()])

        # embeds = torch.cat(embeds, dim=-1)
        # embeds = torch.cat([item.unsqueeze(dim=1) for item in embeds], dim=1)       # [32, 4, 256]

        return Xs[-1], visualize_tools     # [20, 2], 0.75, 0.5, 0.45













