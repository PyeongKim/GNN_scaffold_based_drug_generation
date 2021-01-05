# Author: Pyeong Eun Kim
# Date: Dec.8.2020
# Title: Helper functions for processing data or output of model
# Affiliation: JLK Genome Research Centre

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem import BondType
import torch
from cl_featuriser import AtomFeaturiser, BondFeaturiser
import dgl
from molvs import charge

def dgl_to_mol(G):

    # create mol object
    mol = Chem.RWMol()

    # {atomic_type: atomic number} for nodes in graph G
    atomic_num = {0:6, 1:7, 2:8, 3:9, 4:15, 5:16, 6:17, 7:35}
    aromaticity = {0:False, 1:True}

    for node in G.nodes():
        a=Chem.Atom(atomic_num[G.ndata["atom_type"][node].item()])
        idx = mol.AddAtom(a)

    # {bond_type: rdkit.BondType} for edges in graph G
    bond_types = {
        0:Chem.rdchem.BondType.SINGLE,
        1:Chem.rdchem.BondType.DOUBLE,
        2:Chem.rdchem.BondType.TRIPLE
        }

    # add bonds to mol object
    edges = zip(G.edges()[0].tolist()[1::2],G.edges()[1].tolist()[1::2])
    for j,edge in enumerate(edges):
        first, second = edge
        bond_type = bond_types[G.edata["bond_type"][j*2].item()]
        mol.AddBond(first, second, bond_type)

    mol.UpdatePropertyCache(strict=False)
    pt = Chem.GetPeriodicTable()
    for j,at in enumerate(mol.GetAtoms()):
        atomic_num = at.GetAtomicNum()
        at.SetFormalCharge(0)
        #print(j,at.GetSymbol(), [j.GetSymbol() for j in at.GetNeighbors()] ,[j.GetBondType() for j in at.GetBonds()])#,
        #print(j,at.GetSymbol(), "fc",at.GetFormalCharge(), 'exh',at.GetNumExplicitHs(),'th',at.GetTotalNumHs(),'exv',at.GetExplicitValence(),'tv',at.GetTotalValence(), pt.GetDefaultValence(atomic_num))

        tv, dv = at.GetTotalValence(), pt.GetDefaultValence(atomic_num)
        if tv > dv:
            at.SetNoImplicit(True)
            eh, new_tv = at.GetNumExplicitHs(), at.GetTotalValence()
            if eh > 0:
                if eh-(new_tv - dv) >= 0:
                    at.SetNumExplicitHs(eh-(new_tv - dv))
                else:
                    at.SetNumExplicitHs(0)
                    at.SetFormalCharge((new_tv - dv) - eh)
            elif eh == 0:
                if 0<= new_tv - dv <= 1:
                    at.SetFormalCharge(new_tv - dv)
                else:
                    pass
            #at.SetNoImplicit(False)
        elif tv < dv:
            #at.SetFormalCharge(-(dv-tv))
            at.SetFormalCharge(0)
            at.SetNumExplicitHs(at.GetNumExplicitHs() + dv-tv)
        else:
            pass
        at.SetNumRadicalElectrons(0)
        mol.UpdatePropertyCache(strict=False)
        #print(j,at.GetSymbol(), "fc",at.GetFormalCharge(), 'exh',at.GetNumExplicitHs(),'th',at.GetTotalNumHs(),'exv',at.GetExplicitValence(),'tv',at.GetTotalValence(), pt.GetDefaultValence(atomic_num))

    Chem.rdmolops.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    #print("aa",Chem.MolToSmiles(mol, canonical=True))
    return mol, Chem.MolToSmiles(mol, canonical=True)


def mol_to_graph(mol, canonical_atom_order=True):
    atom_featuriser = AtomFeaturiser()
    bond_featuriser = BondFeaturiser()
    if mol is None:
        print('Invalid mol found')
        return None

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

    g = construct_bigraph_from_mol(mol)

    g.ndata.update(atom_featuriser(mol))

    g.edata.update(bond_featuriser(mol))

    return g

def enum_stereoisomer(molecule, training=True):
    smi = Chem.MolToSmiles(molecule, isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)
    isomers_mol = tuple(EnumerateStereoisomers(mol))
    graph_list = []
    smi_list = []
    for smi in sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers_mol):
        smi_list.append(smi)
        mol = Chem.MolFromSmiles(smi)
        Chem.rdmolops.Kekulize(mol)
        g = mol_to_graph(mol)
        graph_list.append(g)
    if not training:
        return dgl.batch(graph_list), smi_list
    else:
        return dgl.batch(graph_list)

def construct_bigraph_from_mol(mol):

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
