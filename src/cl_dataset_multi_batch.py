# Author: Pyeong Eun Kim
# Date: Dec.4.2020
# Title: Data loader for ScaffoldGNN from cl_graph_multi_batch
# Affiliation: JLK Genome Research Centre

from dataclasses import dataclass, field
from typing import List
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import itertools
import numpy as np
import dgl.backend as F
from enum import Enum, unique
from rdkit.Chem import BondType
import dgl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem import AllChem
from m_helper_functions import dgl_to_mol, enum_stereoisomer
from operator import itemgetter
from molvs import charge

class MolGraphData(Dataset):

    def __init__(self, path_to_df, node_featuriser, edge_featuriser, conditions, normalised=True):
        super(MolGraphData, self).__init__()
        self.whole_df = pd.read_csv(path_to_df)
        self.atom_featuriser = node_featuriser
        self.bond_featuriser = edge_featuriser
        self.conditions = conditions
        self.conditions.extend(["scaffold_"+c for c in conditions])
        #print("condition",conditions, self.conditions)
        self.data_len = self.whole_df.shape[0]
        self.normalised = normalised
        if not normalised:
          mean = {"clogp":3.589836, "tpsa":77.059375, "mw":389.977723}
          std = {"clogp":1.961023, "tpsa":25.281207, "mw":80.655741}
          for c in conditions:
              self.whole_df[[c, "scaffold_"+c]] = (self.whole_df[[c, "scaffold_"+c]] - mean[c])/std[c]

    def __getitem__(self, index):
        single_smi = self.whole_df.loc[index].smiles
        #print(single_smi)
        mol = Chem.MolFromSmiles(single_smi)
        #s = Standardizer()
        #mol = s.standardize(mol)
        g, g_scaffold, action, isomer_target = self.mol_to_graph(mol)
        #print("graph", g)
        _, smiles = dgl_to_mol(g) #test_fix_all
        #print("complete")
        conditions = list(self.whole_df.loc[index][self.conditions])
        return g, g_scaffold, action, conditions, isomer_target, self.whole_df.loc[index].ID, self.smi #test_fix_all

    def construct_bigraph_from_mol(self, mol):

        g = dgl.graph(([], []), idtype=torch.int32)

        # Add nodes
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        # Add edges
        src_list = []
        dst_list = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])

        g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

        return g

    def mol_to_graph(self, mol, canonical_atom_order=False):
        if mol is None:
            print('Invalid mol found')
            return None

        if canonical_atom_order:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        # kekulize molecule
        Chem.rdmolops.Kekulize(mol)
        # get core from whole mol
        core = MurckoScaffold.GetScaffoldForMol(mol)
        #print("core",Chem.MolToSmiles(core), mol.GetSubstructMatches(core))
        #print("mol",Chem.MolToSmiles(mol))
        sub_order = list(mol.GetSubstructMatches(core)[0])
        scaffold_list = sub_order

        for i in range(mol.GetNumAtoms()):
            if i in sub_order:
                continue
            else:
                scaffold_list.append(i)
        mol = Chem.RenumberAtoms(mol, tuple(scaffold_list))

        g = self.construct_bigraph_from_mol(mol)
        g_scaffold = self.construct_bigraph_from_mol(core)


        g.ndata.update(self.atom_featuriser(mol))
        g_scaffold.ndata.update(self.atom_featuriser(core))

        g.edata.update(self.bond_featuriser(mol))
        g_scaffold.edata.update(self.bond_featuriser(core))




        isomer, iso_smi = dgl_to_mol(mol)
        isomers = tuple(EnumerateStereoisomers(isomer))
        smiles = []
        for smi in sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers):
            smiles.append(smi)
        original = Chem.MolToSmiles(mol, isomericSmiles=True)
        try:
            index = smiles.index(original)
            isomer_target = [0]*len(smiles)
            isomer_target[index] = 1
        except:
            isomer_target = [1]*len(smiles)

        actions = []
        src, dest = g.edges()
        for i in range(core.GetNumAtoms(), mol.GetNumAtoms()):
            node_feat = g.ndata["atom_type"][i].unsqueeze(0)
            edge_dests = torch.nonzero(src==i, as_tuple=False).flatten()
            edge_srcs = dest[edge_dests]
            edge_srcs_index = torch.nonzero(edge_srcs<i, as_tuple=False).flatten()
            edge_srcs = edge_srcs[edge_srcs_index]
            edge_feat = g.edata["bond_type"][edge_dests]
            edge_feat = edge_feat[edge_srcs_index]
            #[atom type, edge type, destination]
            if len(edge_srcs) > 0:
                actions.append([
                                node_feat,
                                edge_feat,
                                edge_srcs
                                ])
            elif len(edge_srcs) == 0:
                actions.append([
                                node_feat,
                                torch.tensor([self.bond_featuriser.max_bond_type()]),
                                -1
                                ])
        actions.append([
                        torch.tensor([self.atom_featuriser.max_atom_type()]),
                        -1,
                        -1
                        ])

        return g, g_scaffold, actions, isomer_target

    def __len__(self):
        return self.data_len

    def collate_fn(self, batch):
        node_index, edge_index= [], []
        node_s_index, edge_s_index= [], []
        list_g = []
        list_g_scaffold = []
        list_action = []
        list_condition = []
        list_isomer_target = []
        for i, (g, g_scaffold, action, condition, isomer_target, id, smi) in enumerate(batch): #test_fix_all
            node_index.extend([i]*g.num_nodes())
            edge_index.extend([i]*g.num_edges())
            node_s_index.extend([i]*g_scaffold.num_nodes())
            edge_s_index.extend([i]*g_scaffold.num_edges())
            list_g.append(g)
            list_g_scaffold.append(g_scaffold)
            list_action.append(action)
            list_condition.append(condition)
            list_isomer_target.append(isomer_target)
        return dgl.batch(list_g), dgl.batch(list_g_scaffold), \
               list_action, torch.tensor(list_condition), list_isomer_target, \
               torch.tensor(edge_index),torch.tensor(node_index), \
               torch.tensor(edge_s_index),torch.tensor(node_s_index), id, smi #test_fix_all


class MolGraphDataSampling(Dataset):
    def __init__(self, path_to_df, node_featuriser, edge_featuriser, conditions, normalised=True):
        super(MolGraphTestData, self).__init__()
        self.whole_df = pd.read_csv(path_to_df)
        self.atom_featuriser = node_featuriser
        self.bond_featuriser = edge_featuriser
        self.conditions = conditions
        self.data_len = self.whole_df.shape[0]
        self.normalised = normalised
        if not normalised:
          self.mean = {"clogp":3.589836, "tpsa":77.059375, "mw":389.977723}
          self.std = {"clogp":1.961023, "tpsa":25.281207, "mw":80.655741}

    def __getitem__(self, index):
        single_smi = self.whole_df.loc[index].smiles
        #print(single_smi)
        #mol = Chem.MolFromSmiles(single_smi)
        mol = Chem.MolFromSmiles(single_smi)#, sanitize=False)
        g = self.mol_to_graph(mol)

        mw = Chem.Descriptors.MolWt(mol)
        tpsa = Chem.Descriptors.TPSA(mol)
        logp = Chem.Descriptors.MolLogP(mol)
        if not self.normalised:
            mw = (mw - self.mean["mw"])/self.std["mw"]
            tpsa = (tpsa - self.mean["tpsa"])/self.std["tpsa"]
            logp = (logp - self.mean["logp"])/self.std["logp"]
        conditions = self.whole_df.loc[index][["mw","tpsa","logp"]].to_list()
        conditions.expand([mw, tpsa, logp])

        return g, conditions, single_smi

    def construct_bigraph_from_mol(self, mol):

        g = dgl.graph(([], []), idtype=torch.int32)

        # Add nodes
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        # Add edges
        src_list = []
        dst_list = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])

        g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

        return g

    def mol_to_graph(self, mol, canonical_atom_order=True):

        if mol is None:
            print('Invalid mol found')
            return None

        if canonical_atom_order:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        # kekulize molecule
        Chem.rdmolops.Kekulize(mol)

        g = self.construct_bigraph_from_mol(mol)
        g.ndata.update(self.atom_featuriser(mol))
        g.edata.update(self.bond_featuriser(mol))

        return g

    def __len__(self):
        return self.data_len

    def collate_fn(self, batch):
        list_g_scaffold = []
        list_condition = []
        list_smi = []
        for i, (g_scaffold, condition, smi) in enumerate(batch):
            list_g_scaffold.append(g_scaffold)
            list_condition.append(condition)
            list_smi.append(smi)
        return dgl.batch(list_g_scaffold), torch.tensor(list_condition), list_smi

if __name__=="__main__":
    import time
    from cl_graph_multi_batch import ScaffoldGNN
    from cl_featuriser import AtomFeaturiser, BondFeaturiser
    import torch.nn as nn
    import pandas as pd
    #IPythonConsole.ipython_useSVG=True
    start = time.process_time()
    conditions = ["clogp","mw","tpsa"]
    path_to_df = "/BiO/pekim/GRAPHNET/data/normalised_filtered_data.csv"
    dataset = MolGraphData(path_to_df, AtomFeaturiser(), BondFeaturiser(), conditions)
    end = time.process_time()
    print("time spent to initialise dataset {}".format(end-start))
    leng = dataset.__len__()
    model = ScaffoldGNN(6)
    start = time.process_time()
    ds_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, batch_size=1, shuffle=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = open("../data/error.txt", "w")
    for i, (g, g_scaffold, action, condition, isomer_target, w_index, _,s_index,_, id) in enumerate(ds_loader):
        #print(isomer_target, isomer_target.__len__())
        try:
            _,_,_,a,b,c,d,a1,b1,c1,d1,gl = model(g, g_scaffold, condition, action, isomer_target, w_index, s_index)
        #print(gl[0].ndata["atom_type"], gl[0].num_nodes())
            #print(i,len(d), len(d1))
        #print(d, d1)
            end = time.process_time()
            print("time spent to read dataset {}".format(end-start))
        except:
            writer.write(id + "\n")

    writer.close()
        #out = loss(torch.cat(a, 0),torch.tensor(a1))
        #optimizer.zero_grad()
        #start = time.process_time()
        #out.backward()
        # Update weights
        #optimizer.step()
        #end = time.process_time()
        #print("time spent backprop {}".format(end-start))
        #print("loss {}".format(out))
