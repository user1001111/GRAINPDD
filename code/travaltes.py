import datetime
import os
import copy
import numpy as np
import scipy.sparse as sp
import sys
import time
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score, average_precision_score

from model_MF import MF

from augFeature import getFeature
from utils.torch_data import getLoader

from utils.loss_function import lossF
from utils.clac_metric import get_metrics
from utils.normalization import normalizeRow, normalizeRowCol
from utils.process_set import diagZeroAdj, get_profile_sim
from utils.model_gat import MultiLayerGATFeatureAggregator,preprocess_edge_index



# ========== Tee writes to both the terminal and the log file ==========
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()
# ==============================================================
def getBigAdj(sim_A_total, sim_b_total, traiY):
    adj_net_homo = np.array([np.vstack((np.hstack((sim_A_total[i], np.zeros_like(traiY))),
                                        np.hstack((np.zeros_like(traiY.T), sim_b_total[j]))
                                        )) for i in range(len(sim_A_total)) for j in range(len(sim_b_total))]
                            )
    adj_net_hete = np.array([np.vstack((np.hstack((np.zeros_like(sim_A_total[i]), traiY)),
                                        np.hstack((traiY.T, np.zeros_like(sim_b_total[j])))
                                        )) for i in range(len(sim_A_total)) for j in range(len(sim_b_total))]
                            )

    for i in range(len(adj_net_homo)):
        adj_net_homo[i] = normalizeRowCol(adj_net_homo[i])
        adj_net_hete[i] = normalizeRowCol(adj_net_hete[i])
    return torch.FloatTensor(adj_net_homo), torch.FloatTensor(adj_net_hete)

def testModel(opt, model, loader, F_u, F_i, adj_net_homo, adj_net_hete):
    cudaFlag = opt.cudaFlag;
    device = opt.device
    prin = opt.prin
    lossType = opt.lossType

    model.eval()
    lossLs, y_label, y_pred = [], [], []
    with torch.no_grad():
        for i, (idx0, idx1, y) in enumerate(loader):
            if cudaFlag:
                y = y.to(device)

            output, _ = model(opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete)

            loss = lossF(lossType, output, y, gamma=opt.focal_gamma, alpha=opt.focal_alpha).item()
            lossLs.append(loss)

            y_label += y.cpu().numpy().tolist()
            y_pred += output.detach().cpu().numpy().tolist()
        if prin:
            aveLoss = sum(lossLs) / float(len(lossLs))
            print('average test loss: {:.4f}'.format(aveLoss))


    aupr = average_precision_score(y_label, y_pred)
    auc = roc_auc_score(y_label, y_pred)
    return y_label, y_pred, aupr, auc, loss


def trainModel(opt, model, optimizer, adj_net_homo, adj_net_hete, F_u, F_i, trai_loader, vali_loader, test_loader,
               tes_list):
    # ========== Create the log file and redirect stdout ==========
    log_dir = './training_logs/'
    os.makedirs(log_dir, exist_ok=True)
    kfold = getattr(opt, 'kfold', -1)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}fold_{kfold}_log_{timestamp}.txt"
    tee = Tee(log_filename)
    sys.stdout = tee
    # =======================================================

    epochs = opt.epochs
    lossType = opt.lossType
    earlyFlag = opt.earlyFlag
    patience = opt.patience
    device = opt.device
    prin = opt.prin

    adj_net_homo = adj_net_homo.to(device)
    adj_net_hete = adj_net_hete.to(device)
    F_u = F_u.to(device)
    F_i = F_i.to(device)

    if prin:
        print('Start Training...')
    # Train model
    t_total = time.time()
    # initialization
    model_max = copy.deepcopy(model)
    max_auc = 0
    max_aupr = 0
    loss_history = []
    epoch_patience = 0
    epoch = 0
    epoch_best = 0
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            if prin:
                print('\n-------- Epoch {:04d} --------'.format(epoch))
            y_pred_train = []
            y_label_train = []
            # train epoch
            model.train()
            for i, (idx0, idx1, y) in enumerate(trai_loader):
                idx0 = idx0.to(device)
                idx1 = idx1.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output, E_ls = model(opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete)

                if opt.use_improvements == 1 and opt.label_smoothing > 0:
                    # ---- Dynamic annealing (improved version) ----
                    progress = epoch / epochs
                    if progress < 0.8:
                        current_smooth = opt.label_smoothing * (1 - progress / 0.8)
                    else:
                        current_smooth = 0.0
                    y_target = y * (1 - current_smooth) + 0.5 * current_smooth
                else:
                    # ---- Original: use hard labels directly ----
                    y_target = y

                loss_train = lossF(lossType, output, y_target)  # Original lossF lacks focal loss, although the CLI accepts it.
                # loss_train.backward()
                loss_train.backward(retain_graph=True)
                optimizer.step()
                loss_history.append(loss_train.detach().cpu().item())
                y_label_train += y.cpu().numpy().tolist()
                y_pred_train += output.detach().cpu().numpy().tolist()

            aupr_train = average_precision_score(y_label_train, y_pred_train)
            auc_train = roc_auc_score(y_label_train, y_pred_train)

            epoch_patience += 1
            if epoch_patience > patience:
                print('best epoch is: {:04d} \n early stopping at epoch: {:04d}'.format(epoch_best, epoch))
                break
            # validation after each epoch
            if earlyFlag:
                model.eval()
                y_label_val, y_pred_val, aupr_val, auc_val, loss_val = testModel(opt, model, vali_loader, F_u, F_i,
                                                                                 adj_net_homo, adj_net_hete)
                if aupr_val > max_aupr:
                    max_aupr = aupr_val
                    epoch_patience = 0
                    model_max = copy.deepcopy(model)

                    max_auc = auc_val
                    if prin:
                        print(' epoch:       {:04d}\n'.format(epoch),
                              ' loss_train:  {:.4f}\n'.format(loss_train.item()),
                              ' aupr_train:  {:.4f}\n'.format(aupr_train),
                              ' auroc_train: {:.4f}\n'.format(auc_train),
                              ' loss_val:    {:.4f}\n'.format(loss_val),
                              ' aupr_val:    {:.4f}\n'.format(aupr_val),
                              ' auc_val:     {:.4f}\n'.format(auc_val),
                              ' max_auc:     {:.4f}'.format(max_auc))
            else:
                model_max = copy.deepcopy(model)
            if prin:
                print('the {:04d} epoch take {:.4f} seconds'.format(epoch, time.time() - epoch_start))
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        pass

    if prin:
        print("Optimization Finished!")
        print("Total time elapsed: {:.2f}s".format(time.time() - t_total))
        print('The best epoch is: {:04d} \n current epoch is: {:04d}'.format(epoch_best, epoch))
    # Testing
    model_max.eval()
    test_label, test_score, aupr_test, auc_test, loss_test = testModel(opt, model_max, test_loader, F_u, F_i,
                                                                       adj_net_homo, adj_net_hete)
    if prin:
        print('loss_test: {:.4f}\n'.format(loss_test),
              'aupr_test: {:.4f}\n'.format(aupr_test),
              ' auc_test: {:.4f}'.format(auc_test))

    # ========== Restore stdout and close the log file ==========
    sys.stdout = tee.terminal
    tee.close()
    # =============================================

    return test_label, test_score, model_max



def normalize_features(feat_array, normalizeType):

    if normalizeType == 'col_mean_zero':
        from sklearn.preprocessing import scale
        feat_array = scale(feat_array)
    elif normalizeType == 'minmax':
        from sklearn.preprocessing import minmax_scale
        feat_array = minmax_scale(feat_array)
    elif normalizeType == 'softmax':
        from utils.normalization import normalizeSoft
        feat_array = normalizeSoft(feat_array)
    elif normalizeType == 'row_sum_one':
        from utils.normalization import normalizeRow
        feat_array = normalizeRow(feat_array)
    elif normalizeType == 'none':
        pass
    else:
        raise ValueError(f"Unsupported normalizeType: {normalizeType}")
    return feat_array


def trainTestMain(opt, sim_A, sim_b, tra_list, val_list, tes_list):
    opt.att_dim = opt.mid_dim
    batch_size = opt.batch_size
    lr = opt.lr
    weight_decay = opt.weight_decay
    device = opt.device

    opt.row_num = row_num = sim_A[0].shape[0]
    opt.col_num = col_num = sim_b[0].shape[0]

    trai_loader = getLoader(batch_size, tra_list)
    vali_loader = getLoader(batch_size, val_list)
    test_loader = getLoader(batch_size, tes_list)

    sim_A_total, sim_b_total = copy.deepcopy(sim_A), copy.deepcopy(sim_b)  # Deep copies do not share memory with the originals.

    sim_A_total = get_profile_sim(sim_A_total, tra_list, row_num, col_num)
    tra_T = np.zeros_like(tra_list)
    tra_T[:, 0] = tra_list[:, 1]
    tra_T[:, 1] = tra_list[:, 0]
    tra_T[:, 2] = tra_list[:, 2]
    sim_b_total = get_profile_sim(sim_b_total, tra_T, col_num, row_num)


    sim_A_total = diagZeroAdj(sim_A_total)
    sim_b_total = diagZeroAdj(sim_b_total)

    traiY = sp.coo_matrix((tra_list[:, 2], (tra_list[:, 0], tra_list[:, 1])),
                          shape=(row_num, col_num), dtype=np.float32).toarray()


    adj_net_homo, adj_net_hete = getBigAdj(sim_A_total, sim_b_total, traiY)


    F_u, F_i = getFeature(opt, traiY, sim_A_total, sim_b_total)
    F_u, F_i = torch.FloatTensor(F_u), torch.FloatTensor(F_i)

    F_u, F_i=F_u.to(device),F_i.to(device)

    edge_index_homo = adj_net_homo[0].nonzero(as_tuple=False).t().to(device)
    edge_index_hete = adj_net_hete[0].nonzero(as_tuple=False).t().to(device)

    drug_edge_index = preprocess_edge_index(edge_index_homo, F_u.shape[0]).to(device)
    disease_edge_index = preprocess_edge_index(edge_index_hete, F_i.shape[0]).to(device)

    gat_aggregator_drug = MultiLayerGATFeatureAggregator(
        in_dim=F_u.shape[1],
        hidden_dim=opt.hidden_dim,
        out_dim=F_u.shape[1],
        num_heads=opt.gat_heads,
        dropout=opt.gat_dropout,
        activation=opt.gat_activation,
        num_layers=opt.gat_layers
    ).to(device)

    gat_aggregator_disease = MultiLayerGATFeatureAggregator(
        in_dim=F_i.shape[1],
        hidden_dim=opt.hidden_dim,
        out_dim=F_i.shape[1],
        num_heads=opt.gat_heads,
        dropout=opt.gat_dropout,
        activation=opt.gat_activation,
        num_layers=opt.gat_layers
    ).to(device)

    if opt.use_gat:
        F_u = gat_aggregator_drug(F_u.detach(), drug_edge_index).to(device)
        F_i = gat_aggregator_disease(F_i.detach(), disease_edge_index).to(device)

    opt.num_sim = num_sim = len(sim_A_total) * len(sim_b_total)
    model = MF(opt, num_sim).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    test_label, test_score, model_max = trainModel(opt, model, optimizer, adj_net_homo, adj_net_hete, F_u, F_i,
                                                   trai_loader, vali_loader, test_loader, tes_list)

    criteria_result = get_metrics(np.mat(test_label), np.mat(test_score))
    return test_label, test_score, criteria_result, model_max, F_u, F_i

