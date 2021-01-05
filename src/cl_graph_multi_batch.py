# Author: Pyeong Eun Kim
# Date: Dec.04.2020
# Title: Graph-based Deep Generative Model
# Affiliation: JLK Genome Research Centre

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import dgl
from dgl.readout import readout_nodes
from functools import partial
from m_helper_functions import dgl_to_mol, enum_stereoisomer


class ScaffoldGNN(nn.Module):
    def __init__(self, num_auxiliary, device, num_prop_rounds=[3,3,2,2,2,2],
                 num_node_features=[8,4,9,2], num_edge_features=[3,6],  # if aromatic included [4,5]
                 node_emb_dim=128, edge_emb_dim=128, edge_embedding_per_round=True):
        super(ScaffoldGNN, self).__init__()

        # set number of propagation of each propagating functions
        self.device = device
        self.num_prop_rounds = {"whole":None, "scaffold":None,
                                "addnode":None, "addedge":None,
                                "selectnode":None, "selectisomer":None}
        for n, key in enumerate(self.num_prop_rounds.keys()):
            self.num_prop_rounds[key] = num_prop_rounds[n]

        # set all arguments to be shared
        self.num_auxiliary = num_auxiliary
        self.num_node_types = num_node_features[0]
        self.num_edge_types = num_edge_features[0]
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim

        self.init_node_embedding_atom = nn.Embedding(num_node_features[0], node_emb_dim)
        self.init_node_embedding_chiral = nn.Embedding(num_node_features[1], node_emb_dim)
        self.init_node_embedding_formal_charge = nn.Embedding(num_node_features[2], node_emb_dim)
        self.init_node_embedding_aromatic = nn.Embedding(num_node_features[3], node_emb_dim)

        self.init_edge_embedding_bond = nn.Embedding(num_edge_features[0], edge_emb_dim)
        self.init_edge_embedding_stereo = nn.Embedding(num_edge_features[1], edge_emb_dim)

        """ENCODING PART"""
        # whole graph part
        self.whole_graph_prop = ConditionalProp(self.num_prop_rounds["whole"],
                                                num_auxiliary, node_emb_dim, edge_emb_dim) # node_emb_dim
        self.whole_graph_readout = GraphReadOut(node_embed_dim=node_emb_dim, graph_embed_dim=2*node_emb_dim)  # node_emb_dim*2
        self.latent_dim = node_emb_dim
        self.fc_mu = nn.Linear(node_emb_dim*2, node_emb_dim)
        self.fc_var = nn.Linear(node_emb_dim*2, node_emb_dim)

        """DECODING PART"""
        # scaffold graph part
        self.scaffold_graph_prop = ConditionalProp(self.num_prop_rounds["scaffold"],
                                                   num_auxiliary, node_emb_dim, edge_emb_dim)

        """DECODING ACTION DECISION"""
        # add node
        # we will concat auxiliary with latent vector when propagate scaffold graph
        num_auxiliary_decode = num_auxiliary + node_emb_dim
        self.add_nodes_prop = ConditionalProp(self.num_prop_rounds["addnode"],
                                              num_auxiliary_decode,
                                              node_emb_dim, edge_emb_dim)
        self.add_nodes_readout = GraphReadOut(node_embed_dim=node_emb_dim, graph_embed_dim=node_emb_dim)
        self.add_nodes_mlp = MLP(3, node_emb_dim + num_auxiliary_decode, node_emb_dim, self.num_node_types + 1, True)
        #self.add_nodes_softmax = nn.Softmax(dim=1)

        # init node
        self.init_nodes_mlp1 = nn.Embedding(self.num_node_types, node_emb_dim)
        self.init_nodes_mlp2 = nn.Linear(node_emb_dim*2, node_emb_dim)
        self.init_nodes_readout = GraphReadOut(node_embed_dim=node_emb_dim, graph_embed_dim=node_emb_dim)

        # add edge
        self.add_edges_prop = ConditionalProp(self.num_prop_rounds["addedge"],
                                              num_auxiliary_decode,
                                              node_emb_dim, edge_emb_dim)
        self.add_edges_readout = GraphReadOut(node_embed_dim=node_emb_dim, graph_embed_dim=node_emb_dim)
        # in this case we did not use num_edge_types + 1
        # instead we used num_edge_type because we are not considering adding aromatic bonds
        # thus, the final logit would correspond to [single, double, triple, STOP]
        self.add_edges_mlp = MLP(3, node_emb_dim + num_auxiliary_decode, node_emb_dim, self.num_edge_types + 1, True)
        #self.add_edges_softmax = nn.Softmax(dim=1)

        # init edge
        # use self.num_edge_types -1 because it corresponds to [single, double, triple]
        # from which aromatic is excluded.
        self.init_edges_mlp1 = nn.Embedding(self.num_edge_types, node_emb_dim)
        self.init_edges_mlp2 = nn.Linear(node_emb_dim*2, node_emb_dim)
        self.init_edges_readout = GraphReadOut(node_embed_dim=node_emb_dim, graph_embed_dim=node_emb_dim)

        # select dest node
        self.select_nodes_prop = ConditionalProp(self.num_prop_rounds["selectnode"],
                                                 num_auxiliary_decode,
                                                 node_emb_dim, edge_emb_dim)
        #self.select_nodes_mlp = nn.Linear(node_emb_dim + node_emb_dim + num_auxiliary_decode, 1)
        self.select_nodes_mlp = MLP(3, node_emb_dim + node_emb_dim + num_auxiliary_decode, node_emb_dim, 1, True)

        # select isomers
        self.select_isomer_prop = ConditionalProp(self.num_prop_rounds["selectisomer"],
                                                  num_auxiliary_decode,
                                                  node_emb_dim, edge_emb_dim)
        self.select_isomer_mlp = MLP(3, node_emb_dim + num_auxiliary_decode, node_emb_dim, 1, True)



    def prepare_for_traininig(self):
        self.node_probability = []
        self.edge_probability = []
        self.dest_probability = []
        self.isomer_probability = []
        self.node_probability_gt = []
        self.edge_probability_gt = []
        self.dest_probability_gt = []
        self.isomer_probability_gt = []

    def prepare_for_sampling(self):
        self.selected_isomers = []

    def whole_graph_encode(self, g_whole, conditions, batch_index):
        """
        conditions: real conditions such as MW, logP, etc...
        """
        g_whole.ndata["node_features"] = torch.stack([
                                                self.init_node_embedding_atom(g_whole.ndata["atom_type"]),
                                                self.init_node_embedding_chiral(g_whole.ndata["chirality_type"]),
                                                self.init_node_embedding_formal_charge(g_whole.ndata["charge_type"]),
                                                self.init_node_embedding_aromatic(g_whole.ndata["aromatic_type"]),
                                                ], dim=1).sum(1)
        g_whole.edata["edge_features"] = torch.stack([
                                                self.init_edge_embedding_bond(g_whole.edata["bond_type"]),
                                                self.init_edge_embedding_stereo(g_whole.edata["stereo_type"])
                                                ], dim=1).sum(1)
        g_whole = self.whole_graph_prop(g_whole, conditions,batch_index)
        # print("################after whole graph:", g_whole.edata["edge_features"].size())
        readout = self.whole_graph_readout(g_whole)
        mu = self.fc_mu(readout)
        logvar = self.fc_var(readout)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z, logvar, mu

    def scaffold_propagate(self, g_scaffold, conditions, batch_index):
        g_scaffold.ndata["node_features"] = torch.stack([
                                                self.init_node_embedding_atom(g_scaffold.ndata["atom_type"]),
                                                self.init_node_embedding_chiral(g_scaffold.ndata["chirality_type"]),
                                                self.init_node_embedding_formal_charge(g_scaffold.ndata["charge_type"]),
                                                self.init_node_embedding_aromatic(g_scaffold.ndata["aromatic_type"]),
                                                ], dim=1).sum(1)
        g_scaffold.edata["edge_features"] = torch.stack([
                                                self.init_edge_embedding_bond(g_scaffold.edata["bond_type"]),
                                                self.init_edge_embedding_stereo(g_scaffold.edata["stereo_type"])
                                                ], dim=1).sum(1)
        g_scaffold = self.scaffold_graph_prop(g_scaffold, conditions, batch_index)

        return g_scaffold

    def add_nodes_encode(self, g_scaffold, conditions, batch_index, training=True):
        g_scaffold = self.add_nodes_prop(g_scaffold, conditions, batch_index)
        readout = self.add_nodes_readout(g_scaffold)
        readout = torch.cat([readout, conditions], dim=1)
        readout = self.add_nodes_mlp(readout)
        #readout = self.add_nodes_softmax(readout)
        if not training:
            # instead of sampling max, we sampled from distribution
            dist = F.softmax(readout, dim=0)
            dist = Categorical(dist)
            return dist.sample()
        else:
            return readout

    def init_nodes_encode(self, g_scaffold, categorical_node_feat):
        #print("node_feat",categorical_node_feat)
        node_feat = self.init_nodes_mlp1(categorical_node_feat) #Gxembed
        readout = self.init_nodes_readout(g_scaffold)   # Gxembed
        readout = torch.cat([readout, node_feat], dim=1)
        node_feat = self.init_nodes_mlp2(readout)
        return node_feat

    def add_edges_encode(self, g_scaffold, conditions, batch_index,training=True):
        g_scaffold = self.add_edges_prop(g_scaffold, conditions, batch_index)
        readout = self.add_edges_readout(g_scaffold)
        readout = torch.cat([readout, conditions], dim=1)
        readout = self.add_edges_mlp(readout)
        #readout = self.add_edges_softmax(readout)
        if not training:
            # instead of sampling max, we sampled from distribution
            dist = F.softmax(readout, dim=0)
            dist = Categorical(dist)
            return dist.sample()
        else:
            return readout

    def init_edges_encode(self, g_scaffold, categorical_edge_feat):
        edge_feat = self.init_edges_mlp1(categorical_edge_feat) #Gxembed
        readout = self.init_edges_readout(g_scaffold)   # Gxembed
        readout = torch.cat([readout, edge_feat], dim=1)
        edge_feat = self.init_edges_mlp2(readout)
        return edge_feat

    def select_nodes_encode(self, g_scaffold, conditions, batch_index, training=True):
        src = g_scaffold.number_of_nodes() - 1
        possible_dest = range(src)
        g_scaffold = self.select_nodes_prop(g_scaffold, conditions, batch_index)
        possible_dest_emb = g_scaffold.nodes[possible_dest].data["node_features"]
        src_emb_expand = g_scaffold.nodes[src].data["node_features"].expand(src, -1)
        condition_expand = conditions.expand(src, -1)
        # print(possible_dest_emb.size(), src_emb_expand.size(), condition_expand.size())
        logits = self.select_nodes_mlp(torch.cat([possible_dest_emb,
                                                  src_emb_expand,
                                                  condition_expand], dim=1).float()).squeeze(1)
        if not training:
            dist = F.softmax(logits, dim=0)
            dist = Categorical(dist)
            return dist.sample()
        else:
            return logits

    """
    def select_isomer_encode(self, g, condition, training=True):
        mol, smiles = dgl_to_mol(g)  #test_fix_all
        assert mol is not None, "Not valid molecule"
        #condition = conditions[index].unsqueeze(0)
        if not training:
            isomers, isomers_smi = enum_stereoisomer(mol, training)  # batched graphs
        else:
            isomers = enum_stereoisomer(mol, training)
        isomers.ndata["node_features"] = torch.stack([
                                                self.init_node_embedding_atom(isomers.ndata["atom_type"]),
                                                self.init_node_embedding_chiral(isomers.ndata["chirality_type"]),
                                                self.init_node_embedding_formal_charge(isomers.ndata["charge_type"]),
                                                self.init_node_embedding_aromatic(isomers.ndata["aromatic_type"]),
                                                ], dim=1).sum(1)
        isomers.edata["edge_features"] = torch.stack([
                                                self.init_edge_embedding_bond(isomers.edata["bond_type"]),
                                                self.init_edge_embedding_stereo(isomers.edata["stereo_type"])
                                                ], dim=1).sum(1)
        isomers = self.select_isomer_prop(isomers, condition)
        readout = readout_nodes(isomers,"node_features", op="mean")
        readout = torch.cat([readout, condition.expand(readout.size(0), -1)], dim=1)
        logits = self.select_isomer_mlp(readout)  # Nx1
        #print(logits.size())
        if not training:
            dist = F.softmax(logits, dim=0)
            dist = Categorical(dist)
            return isomers_smi[dist.sample().item()]
        else:
            return isomers, logits, smiles #test_fix_all
    """


    def forward(self, g_whole, g_scaffold, conditions, actions,  batch_index, sc_index):
        self.prepare_for_traininig()
        z, logvar, mu = self.whole_graph_encode(g_whole, conditions, batch_index)
        g_scaffold = self.scaffold_propagate(g_scaffold, conditions, sc_index)
        # print("after scaffold:",g_scaffold.edata["edge_features"].size())
        # concat conditions with latent vector for decoding process
        conditions = torch.cat([z, conditions], dim=1)
        #print("!!!!!!!!!!!!!!!!!!!!conditions", conditions.size())
        g_scaffold_list = dgl.unbatch(g_scaffold)
        for i in range(len(g_scaffold_list)):
            sub_graph, sub_action, sub_condition = \
                                            g_scaffold_list[i], actions[i], conditions[i].unsqueeze(0)
            #print(sub_graph.ndata["explicit_hydrogen"])
            add_node = True
            index = 0
            node_prop, edge_prop, dest_prop = [],[],[]
            node_prop_gt, edge_prop_gt, dest_prop_gt = [],[],[]
            while add_node:
                # add node but which node?
                node_dist = self.add_nodes_encode(sub_graph, sub_condition, None)  # if stop add_node = False (setting from dataset)
                # print("node size",node_dist.size())  # (batch x atom_type + 1)
                #self.node_probability.append(node_dist)
                self.node_probability.append(node_dist)
                self.node_probability_gt.append(sub_action[index][0].item())
                if sub_action[index][0].item() == self.num_node_types:
                    add_node = False
                    continue
                else:
                    num_nodes = sub_graph.number_of_nodes()
                    # get node feature
                    init_node = sub_action[index][0].to(self.device)
                    new_node_feat = self.init_nodes_encode(sub_graph, init_node)
                    sub_graph.add_nodes(1)
                    # print(new_node_feat.size(), g_scaffold.nodes[num_nodes - 1].data['node_features'].size())
                    sub_graph.nodes[num_nodes].data['node_features'] = new_node_feat
                    sub_graph.nodes[num_nodes].data["atom_type"] = init_node
                edge_index = 0
                add_edge = True
                #print(actions[index], actions)
                while add_edge and edge_index < len(sub_action[index][1]):
                    edge_add_action = self.add_edges_encode(sub_graph, sub_condition, None)
                    self.edge_probability.append(edge_add_action)
                    self.edge_probability_gt.append(sub_action[index][1][edge_index])
                    # print("%%%%%%%%%%%",actions[index], edge_index)
                    if sub_action[index][1][edge_index] == self.num_edge_types:
                        #print("!!!!!!!!!!!!!!{}!!!!!!!!!!!!".format(sub_action[index][1][edge_index]))
                        add_edge = False
                        continue
                    else:
                        init_edge = sub_action[index][1][edge_index].unsqueeze(0).to(self.device)  # edge types
                        # print("after scaffold:",g_scaffold.edata["edge_features"].size())
                        select_node = self.select_nodes_encode(sub_graph, sub_condition, None)
                        # print("###############after scaffold:",g_scaffold.edata["edge_features"].size())
                        self.dest_probability.append(select_node)
                        self.dest_probability_gt.append(sub_action[index][2][edge_index])
                        src, dest = num_nodes, sub_action[index][2][edge_index]  # edge dest
                        src_list = [src, dest]
                        dest_list = [dest, src]
                        sub_graph.add_edges(src_list, dest_list)
                        # print(init_edge)
                        edge_emb = self.init_edges_encode(sub_graph, init_edge)
                        # print("###edge###",sub_graph.edges[src_list, dest_list].data['edge_features'].size(),torch.cat([edge_emb, edge_emb], dim=0).size())
                        sub_graph.edges[src_list, dest_list].data['edge_features'] = torch.cat([edge_emb, edge_emb], dim=0)
                        sub_graph.edges[src_list, dest_list].data['bond_type'] = torch.cat([init_edge, init_edge], dim=0)
                    edge_index += 1
                index += 1
            g_scaffold_list[i] = sub_graph


        return z, logvar, mu, \
               self.node_probability, self.edge_probability, self.dest_probability,\
               self.node_probability_gt, self.edge_probability_gt, self.dest_probability_gt,\
               g_scaffold_list


    def sample(self, num_samples, g_scaffold_batch, conditions_batch, smi_batch, save_file):
        g_scaffold_batch_list = dgl.unbatch(g_scaffold_batch)
        csv_file = open(save_file, "w")
        for batch, g_scaffold in enumerate(g_scaffold_batch_list):
            self.prepare_for_sampling()
            z = torch.randn(num_samples,
                            self.latent_dim)
            g_scaffold = dgl.batch([g_scaffold for i in num_samples])
            conditions = condtions_batch[batch]
            conditions = conditions.expand(num_samples, -1)
            g_scaffold = self.scaffold_propagate(g_scaffold, conditions, None)
            # concat conditions with latent vector for decoding process
            conditions = torch.cat([z, conditions], dim=1)
            g_scaffold_list = dgl.unbatch(g_scaffold)
            for i in range(num_samples):
                sub_graph, sub_condition = g_scaffold_list[i], conditions[i].unsqueeze(0)
                add_node = True
                index = 0
                while add_node:
                    # add node but which node?
                    node_dist = self.add_nodes_encode(sub_graph, sub_condition, None, False)  # if stop add_node = False (setting from dataset)
                    node_dist = node_dist.item()
                    if node_dist == self.num_node_types:
                        add_node = False
                        continue
                    else:
                        num_nodes = sub_graph.number_of_nodes()
                        # get node feature
                        new_node_feat = self.init_nodes_encode(sub_graph, node_dist)
                        sub_graph.add_nodes(1)
                        sub_graph.nodes[num_nodes].data['node_features'] = new_node_feat
                        sub_graph.nodes[num_nodes].data["atom_type"] = node_dist

                    edge_index = 0
                    add_edge = True
                    while add_edge:
                        selected_edge = self.add_edges_encode(sub_graph, sub_condition, None, False)
                        selected_edge = selected_edge.item()
                        if selected_edge == self.num_edge_types:
                            add_edge = False
                            continue
                        else:
                            # print("after scaffold:",g_scaffold.edata["edge_features"].size())
                            selected_node = self.select_nodes_encode(sub_graph, sub_condition, None, False)
                            # print("###############after scaffold:",g_scaffold.edata["edge_features"].size())
                            src, dest = num_nodes, selected_node  # edge dest
                            src_list = [src, dest]
                            dest_list = [dest, src]
                            sub_graph.add_edges(src_list, dest_list)
                            # print(init_edge)
                            edge_emb = self.init_edges_encode(sub_graph, selected_edge)
                            # print("###edge###",sub_graph.edges[src_list, dest_list].data['edge_features'].size(),torch.cat([edge_emb, edge_emb], dim=0).size())
                            sub_graph.edges[src_list, dest_list].data['edge_features'] = torch.cat([edge_emb, edge_emb], dim=0)
                            sub_graph.edges[src_list, dest_list].data['bond_type'] = torch.cat([selected_edge, selected_edge], dim=0)
                        edge_index += 1
                    index += 1



class ConditionalProp(nn.Module):
    def __init__(self, num_prop_rounds, num_auxiliary,
                 node_emb_dim=128, edge_emb_dim=128):
        super(ConditionalProp, self).__init__()

        # number of propagation
        self.num_prop_rounds = num_prop_rounds
        # print("auxiliary number:", num_auxiliary)
        # make propagation layers corresponding to num rounds
        self.prop_layers = nn.ModuleList()
        for round in range(num_prop_rounds):
            self.prop_layers.append(SingleProp(num_auxiliary, node_emb_dim, edge_emb_dim))

    def forward(self, g, auxiliary, batch_index=None):
        # condition should be list of conditions
        # node categorical to embed
        """
        # if feed g of whole molecule and scaffold, "node_features, and edge_features"
        # should be already embedded by shared embedding layer
        """
        # print("**********edge feature")
        # print(g.edata["edge_features"].size())
        original_edge = g.edata["edge_features"]
        if batch_index is not None: # if batch > 1
            auxiliary_expand = auxiliary[batch_index]
        else:
            auxiliary_expand = auxiliary.expand(g.num_edges(),-1)
        #print(original_edge.size(), auxiliary_expand.size(), batch_index.size(), auxiliary.size())
        #torch.cat((g.edata["edge_features"], auxiliary.expand(g.num_edges(),-1)), -1)
        g.edata["edge_features"] = torch.cat((original_edge, auxiliary_expand), 1)
        for round in range(self.num_prop_rounds):
            g = self.prop_layers[round](g)
        g.edata["edge_features"] = original_edge
        return g

class MLP(nn.Module):
    """ MultilayerPerceptron (modified from https://github.com/weihua916/powerful-gnns/blob/master/models/mlp.py)
    Args:
        num_layers (int): number of layers in the neural networks. If num_layers=1, this reduces to linear model.
        input_dim (int): dimensionality of input features
        hidden_dim (int): dimensionality of hidden units at ALL layers
        output_dim (int): number of classes for prediction
        batch_norm (Bool): batch normalisation of after MLP (default to True)
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, batch_norm=False):
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                else:
                    self.batch_norms.append(nn.Identity())
    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x.float()
            for layer in range(self.num_layers - 1):
                # print(layer, h.size(), self.linears[layer])
                ## print(self.linears[layer](h))
                if len(h.size()) > 2:
                    h = F.relu(self.linears[layer](h))
                else:

                    h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

class SingleProp(nn.Module):
    def __init__(self, num_auxiliary, node_emb_dim,
                 edge_emb_dim=None, num_mlp_layer=2,
                 activation_dim=None, hidden_dim=None,
                 batch_norm=True):
        super(SingleProp, self).__init__()
        # num_auxiliary can be number of conditions or dimension of latent vector
        # eg) 4 for conditions such as [logP_whole, mw_whole, logP_scaffold, mw_whole]
        # eg) 128 for latent vectors such as [0.1, 0.3, ..., 0.9, -1.1]
        self.num_auxiliary = num_auxiliary

        self.node_emb_dim = node_emb_dim

        if not edge_emb_dim:
            edge_emb_dim = node_emb_dim
        if not hidden_dim:
            # same as GIN paper
            hidden_dim = 2 * (node_emb_dim*2 + edge_emb_dim + self.num_auxiliary)

        # if custom activation dim exists apply; otherwise use default
        if activation_dim:
            self.activation_dim = activation_dim
        else:
            # same as paper <https://arxiv.org/abs/1905.13639> indicated
            self.activation_dim = node_emb_dim

        # input being [hv, hu, xuv]
        # dimension [node_hidden_dim, node_hidden_dim, edge_hidden_dim]

        self.message_func = MLP(num_layers=num_mlp_layer,
                                input_dim=node_emb_dim*2 + edge_emb_dim + self.num_auxiliary,
                                hidden_dim=hidden_dim,
                                output_dim=self.activation_dim,
                                batch_norm=False)
        self.reduce_func = partial(self.get_activation_node)
        self.node_update_func = nn.GRUCell(self.activation_dim, node_emb_dim)

        # initialise model parameters (Xaiver Uniform)
        #self.initialise_parameters()

    def initialise_parameters(self):
        """Reinitialise model parameters."""

        nn.init.xavier_uniform_(self.node_update_func.weight.data)
        nn.init.xavier_uniform_(self.message_func)
        if self.edge_embedding_per_round:
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def concat_msg(self, edges):
        """
            Concatenate features from source nodes with edge features and aggregate them in destination nodes.
            For an edge u->v, return concat([h_u, x_uv]).
            v is the destination node which would get the aggregated signals.
        """
        return {'m': torch.cat([edges.src['node_features'],  # feature of neighbor nodes
                                edges.data['edge_features']], # feature of edges connecting to neighbor nodes
                                dim=1)}

    def get_activation_node(self, nodes):
        """
            Concatenate features of each node with features of its neighboring nodes and edges connecting the node.
            Condition is already concatenated within the edge features.
            Then, we use MLP to calculate weighted sum of all messages, which is called activation of nodes.
        """
        node_old = nodes.data['node_features']
        m = nodes.mailbox['m'] # collect messages assigned by msg which is fed to update_all function.
        message = torch.cat([node_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        # print("message_size()--------------------")
        # print(message.size(), m.size(), node_old.size())
        # print("--------------------")
        node_activation = self.message_func(message).sum(1)
        return {'a': node_activation}


    def forward(self, g):

        #edge_condition = torch.cat((g.edata["edge_features"], auxiliary.expand(g.num_edges(),-1)), -1)
        #g.edata["edge_features"] = self.message_func(edge_condition).sum(1)
        #g.edata["edge_features"] = torch.cat((g.edata["edge_features"], auxiliary.expand(g.num_edges(),-1)), -1)
        # print("feat size()--------------------")
        # print(g.edata["edge_features"].size(), auxiliary)
        # print("--------------------")
        # get activation of each nodes ("a")
        g.update_all(message_func=self.concat_msg,
                     reduce_func=self.get_activation_node)

        # update node features based on the activation of nodes
        g.ndata['node_features'] = self.node_update_func(g.ndata['a'], g.ndata['node_features'])

        return g

class GraphReadOut(nn.Module):
    def __init__(self, node_embed_dim, graph_embed_dim=None, activation="sigmoid", mode='mean'):
        super(GraphReadOut, self).__init__()

        assert mode in ['max', 'mean', 'sum'], \
            "Expect mode to be 'max' or 'mean' or 'sum', got {}".format(mode)
        assert activation in ['sigmoid', 'relu', 'tanh'], \
            "Expect activation to be 'sigmoid' or 'relu' or 'tanh', got {}".format(activation)
        self.mode = mode
        # Setting from the paper
        if graph_embed_dim:
            self.graph_embed_dim = graph_embed_dim
        else:
            self.graph_embed_dim = 2 * node_embed_dim

        if activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_embed_dim, 1),
            act
        )
        self.node_to_graph = nn.Linear(node_embed_dim,
                                       self.graph_embed_dim)

    def forward(self, g):
        node_embeds = g.ndata['node_features']
        graph_embeds = self.node_gating(node_embeds) * self.node_to_graph(node_embeds)
        with g.local_scope():
            g.ndata['graph_features'] = graph_embeds
            if self.mode == 'max':
                graph_feats = dgl.max_nodes(g, 'graph_features')
            elif self.mode == 'mean':
                graph_feats = dgl.mean_nodes(g, 'graph_features')
            elif self.mode == 'sum':
                graph_feats = dgl.sum_nodes(g, 'graph_features')
        return graph_feats
