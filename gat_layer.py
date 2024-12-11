import torch
from torch import nn


class GATLayer(nn.Module):
    # implements GAT (multi-head self-attention)
    def __init__(self, in_dim, out_dim, num_heads, activation, concat, device):
        super(GATLayer, self).__init__()
        # whether to run on cpu or gpu
        self.device = device
        # whether to concatenate or average single heads results
        self.concat = concat
        # number of attention heads
        self.num_heads = num_heads
        # output dimension
        self.out_dim = out_dim
        # learnable matrix transforming node embeddings
        self.W = nn.Linear(in_dim, num_heads*out_dim, bias=False)
        # learnable vector of attention coefficients
        self.W_att = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.leakyReLU = nn.LeakyReLU(0.2)  # as in original GAT
        self.softmax = nn.Softmax(dim=1) #maybe 1?
        # non-linearity for current layer
        self.activation = activation
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_att)

    def forward(self, h_i_list, h_j_list):
        b_size = h_i_list.shape[0]
        proj_neigh = self.W(h_j_list).reshape(b_size, -1, self.num_heads, self.out_dim)
        proj_self = self.W(h_i_list).reshape(b_size, -1, self.num_heads, self.out_dim)
        proj_cat = proj_neigh + proj_self
        e_ij = self.leakyReLU((proj_cat * self.W_att).sum(dim=-1))
        alpha_ij = self.softmax(e_ij).view(b_size, -1, self.num_heads, 1)
        alpha_rep = alpha_ij.repeat_interleave(self.out_dim, 3)
        weight_mul = alpha_rep * proj_neigh.view(b_size, -1, self.num_heads, self.out_dim)
        weight_sum = torch.sum(weight_mul.clone(), dim=1)  # [#batch, #heads, #dim]
        if self.concat:
            weight_sum = weight_sum.reshape(b_size, 1, self.num_heads * self.out_dim)
        else:
            weight_sum = weight_sum.mean(dim=1).reshape(b_size, 1, -1)
        h_new = weight_sum.clone()
        if self.activation:
            h_new = self.activation(weight_sum)
        return h_new.to(self.device)




class GAT(nn.Module):
    def __init__(self, num_layers, num_heads_layer_list, num_features_layer_list, device, reverse=False):
        super(GAT, self).__init__()
        num_heads_layer = [1] + num_heads_layer_list
        self.gat_layers = nn.ModuleList()

        for i in range(num_layers):
            #for forward pass, the first layer takes `v_types` as input, subsequent layers take `hidden_size`.
            #for reverse pass, all layers take `hidden_size` as input since they operate on hidden representations.
            if reverse and i == 0:
                in_dim = num_features_layer_list[1]  # hidden_size (reverse first layer processes hidden representation)
            else:
                in_dim = num_features_layer_list[i] * num_heads_layer[i]  # Raw features (forward first layer) or hidden size

            out_dim = num_features_layer_list[i+1]  # This is the hidden_size for both forward and reverse passes

            layer = GATLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                num_heads=num_heads_layer[i+1],
                activation=nn.ReLU() if i < num_layers - 1 else None,
                concat=True if i < num_layers - 1 else False,
                device=device
            )
            self.gat_layers.append(layer)
