
import random
import time
from args import parameter_parser
from util.loadMatData import generate_partition
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from IMPRLNet import IMPRLNet_classify
from util.loadMatData import features_to_Lap, load_data_2, get_evaluation_results


def train(idx_train,idx_test,n, feature_list, lap_list, labels, n_feats, n_view, n_classes,args, device):
    repnum = 10
    all_ACC = []
    all_Fmi = []
    all_Fma = []
    begin_time = time.time()
    labels = torch.from_numpy(labels).long().to(device)

    phi = np.ones(n) - np.identity(n)

    for times in range(repnum):
        best_ACC = 0
        best_F1_macro = 0
        best_F1_micro = 0
        # network architecture
        model = IMPRLNet_classify(n_feats, n_view, n_classes, args, phi, args.block, device).to(device)
        loss_function = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        with tqdm(total=args.epoch, desc="Training") as pbar:
            for epoch in range(args.epoch):
                model.train()
                Z = model(feature_list, lap_list)

                output = Z[-1]
                output = F.log_softmax(output, dim=1)
                loss_sup = loss_function(output[idx_train], labels[idx_train])

                loss = args.gamma * loss_sup
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    Z = model(feature_list, lap_list)
                    output = Z[-1]

                    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                    ACC, F1_macro, F1_micro = get_evaluation_results(labels.cpu().detach().numpy()[idx_test],
                                                                     pred_labels[idx_test])

                    if ACC >= best_ACC:
                        best_ACC = ACC
                        best_F1_macro = F1_macro
                        best_F1_micro = F1_micro
                        # pbar.set_postfix
                    print({'Loss': '{:.6f}'.format(loss.item()),
                           'ACC': '{:.2f} | {:.2f}'.format(ACC * 100, best_ACC * 100),
                           'F1_macro': '{:.2f} | {:.2f}'.format(F1_macro * 100, best_F1_macro * 100),
                           'F1_micro': '{:.2f} | {:.2f}'.format(F1_micro * 100, best_F1_micro * 100)})
                pbar.update(1)
            print('\n')
            print('times = ', times)
            print("------------------------")
            print("ACC:   {:.2f}".format(best_ACC * 100))
            # print("Std:   {:.2f}".format(* 100))
            print("F1_macro:   {:.2f}".format(best_F1_macro * 100))
            print("F1_micro:   {:.2f}".format(best_F1_micro * 100))
        # draw_plt(output, labels, dataset)
        all_ACC.append(best_ACC)
        all_Fma.append(best_F1_macro)
        all_Fmi.append(best_F1_micro)

    cost_time = time.time() - begin_time
    print("------------------------")
    print("ACC:   {:.2f}".format(np.mean(all_ACC) * 100))
    print("Std:   {:.2f}".format(np.std(all_ACC) * 100))
    print("F1_macro:   {:.2f}".format(np.mean(all_Fma) * 100))
    print("F1_micro:   {:.2f}".format(np.mean(all_Fmi) * 100))
    print("cost_time:  {:.2f}".format(cost_time))
    print("------------------------")

if __name__ == '__main__':

    args = parameter_parser()
    args.device = 'cpu'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    dataset = "HW"
    print("load {} dataset...".format(dataset))
    feature_list, labels = load_data_2(dataset, args.data_path)
    n_view = len(feature_list)
    n_feats = [x.shape[1] for x in feature_list]
    n_classes = len(np.unique(labels))
    n = feature_list[0].shape[0]
    print("samples:{}, view size:{}, feature dimensions:{}, class:{}".format(n, n_view, n_feats, n_classes))
    print("construct Laplace matrix...")
    lap_list, adj_list = features_to_Lap(feature_list, 5)

    for i in range(n_view):
        exec("feature_list[{}] = torch.from_numpy(feature_list[{}]/1.0).float().to(device)".format(i, i))
        exec("lap_list[{}] = lap_list[{}].to_dense().to(device)".format(i, i))

    print("*" * 40 + "\nfusion:{},active:{},gamma:{},block:{},epoch:{},thre:{},lr:{},beta:{},delta:{}\n".format(
        args.fusion_type, args.active, args.gamma, args.block, args.epoch, args.thre, args.lr, args.beta, args.delta))

    idx_labeled, idx_unlabeled = generate_partition(labels=labels, ratio=args.ratio)
    # fix the random seed
    if args.fix_seed:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # training
    train(idx_labeled, idx_unlabeled, n, feature_list, lap_list, labels, n_feats, n_view, n_classes,
               args,
               device)
