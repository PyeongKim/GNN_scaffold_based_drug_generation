import time
from cl_graph_multi_batch import ScaffoldGNN
from cl_featuriser import AtomFeaturiser, BondFeaturiser
from cl_dataset_multi_batch import *
import torch.nn as nn
import pandas as pd
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
##torch.set_num_threads(32)
def main():
    conditions = ["clogp","mw","tpsa"]
    path_to_df = "/BiO/pekim/GRAPHNET/data/small.csv"
    device = torch.device("cuda:0")
    dataset = MolGraphData(path_to_df, AtomFeaturiser(), BondFeaturiser(), conditions)
    batch_size = 32
    lr = 0.005
    model = ScaffoldGNN(6, device)
    """if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()"""
    model.to(device)
    ds_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=dataset.collate_fn,
                                            batch_size=batch_size, num_workers=32,shuffle=True)
    loss_n = nn.CrossEntropyLoss()
    loss_e = nn.CrossEntropyLoss()
    loss_d = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    print("number of dataset: {}".format(dataset.__len__()))
    print("training initialized with batch size of {}, learning rate of {}".format(batch_size, lr))
    for e in range(100):
        for i, (g, g_scaffold, action, condition, w_index, _,s_index,_) in enumerate(ds_loader):
            #print(isomer_target, isomer_target.__len__())
            start = time.process_time()
            g, g_scaffold, condition, action, w_index, s_index = g.to(device), g_scaffold.to(device), condition.to(device), action, w_index.to(device), s_index.to(device)
            z, logvar, mu, node_prop, edge_prop, dest_prop,node_prop_gt,edge_prop_gt,dest_prop_gt, graph_list = model(g, g_scaffold, condition, action, w_index, s_index)
            end = time.process_time()
            print("time spent to forward dataset {}".format(end-start))
            #print("node", torch.cat(node_prop, 0).size(),torch.tensor(node_prop_gt).size())
            #print("edge", torch.cat(edge_prop, 0).size(),torch.tensor(edge_prop_gt).size())
            #print("dest", torch.cat(dest_prop, 0).size(),torch.tensor(dest_prop_gt).size())
            loss_node = loss_n(torch.cat(node_prop, 0),torch.tensor(node_prop_gt).to(device))
            loss_edge = loss_e(torch.cat(edge_prop, 0),torch.tensor(edge_prop_gt).to(device))
            loss_dest = torch.tensor([]).to(device)
            for j,dest in enumerate(dest_prop):
                #print(torch.tensor(dest).unsqueeze(0), torch.tensor(dest_prop_gt[j], dtype=torch.float).unsqueeze(0))
                batch_loss = loss_d(dest.unsqueeze(0), torch.as_tensor(dest_prop_gt[j]).unsqueeze(0).long().to(device))
                loss_dest = torch.cat([loss_dest, batch_loss], dim=0)
                #print(batch_loss)
            loss_dest = torch.mean(loss_dest)
            #print(loss_dest)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            loss = 1*loss_node + 1*loss_edge + 1*loss_dest + 1*kld_loss
            print("epoch: {}({}), total loss: {:.3f}, n_l: {:.3f}, e_l: {:.3f}, d_l {:.3f}, kdl: {:.3f}".format(e, i, loss,loss_node, loss_edge , loss_dest , kld_loss))
            #print("loss {}".format(loss))
            optimizer.zero_grad()

            start = time.process_time()
            loss.backward()
            # Update weights
            optimizer.step()
            #scheduler.step()
            end = time.process_time()
            print("time spent backprop {}".format(end-start))

def test_main():
    conditions = ["clogp","mw","tpsa"]
    path_to_df = "/BiO/pekim/GRAPHNET/data/small.pkl"
    whole_bin = "/BiO/pekim/GRAPHNET/data/whole_small.bin"
    scaffold_bin = "/BiO/pekim/GRAPHNET/data/scaffold_small.bin"
    device = torch.device("cuda:0")
    dataset = MolGraphDataTest(path_to_df, whole_bin, scaffold_bin, conditions)
    batch_size = 32
    lr = 0.005
    model = ScaffoldGNN(6, device)
    """if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()"""
    model.to(device)
    ds_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=dataset.collate_fn,
                                            batch_size=batch_size, num_workers=32,shuffle=True)
    loss_n = nn.CrossEntropyLoss()
    loss_e = nn.CrossEntropyLoss()
    loss_d = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    print("number of dataset: {}".format(dataset.__len__()))
    print("training initialized with batch size of {}, learning rate of {}".format(batch_size, lr))
    for e in range(100):
        for i, (g, g_scaffold, action, condition, w_index, _,s_index,_) in enumerate(ds_loader):
            #print(isomer_target, isomer_target.__len__())
            start = time.process_time()
            g, g_scaffold, condition, action, w_index, s_index = g.to(device), g_scaffold.to(device), condition.to(device), action, w_index.to(device), s_index.to(device)
            z, logvar, mu, node_prop, edge_prop, dest_prop,node_prop_gt,edge_prop_gt,dest_prop_gt, graph_list = model(g, g_scaffold, condition, action, w_index, s_index)
            end = time.process_time()
            print("time spent to forward dataset {}".format(end-start))
            #print("node", torch.cat(node_prop, 0).size(),torch.tensor(node_prop_gt).size())
            #print("edge", torch.cat(edge_prop, 0).size(),torch.tensor(edge_prop_gt).size())
            #print("dest", torch.cat(dest_prop, 0).size(),torch.tensor(dest_prop_gt).size())
            loss_node = loss_n(torch.cat(node_prop, 0),torch.tensor(node_prop_gt).to(device))
            loss_edge = loss_e(torch.cat(edge_prop, 0),torch.tensor(edge_prop_gt).to(device))
            loss_dest = torch.tensor([]).to(device)
            for j,dest in enumerate(dest_prop):
                #print(torch.tensor(dest).unsqueeze(0), torch.tensor(dest_prop_gt[j], dtype=torch.float).unsqueeze(0))
                batch_loss = loss_d(dest.unsqueeze(0), torch.as_tensor(dest_prop_gt[j]).unsqueeze(0).long().to(device))
                loss_dest = torch.cat([loss_dest, batch_loss], dim=0)
                #print(batch_loss)
            loss_dest = torch.mean(loss_dest)
            #print(loss_dest)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            loss = 1*loss_node + 1*loss_edge + 1*loss_dest + 1*kld_loss
            print("epoch: {}({}), total loss: {:.3f}, n_l: {:.3f}, e_l: {:.3f}, d_l {:.3f}, kdl: {:.3f}".format(e, i, loss,loss_node, loss_edge , loss_dest , kld_loss))
            #print("loss {}".format(loss))
            optimizer.zero_grad()

            start = time.process_time()
            loss.backward()
            # Update weights
            optimizer.step()
            scheduler.step()
            end = time.process_time()
            print("time spent backprop {}".format(end-start))

if __name__=="__main__":
    #print(torch.multiprocessing.get_sharing_strategy())
    test_main()
