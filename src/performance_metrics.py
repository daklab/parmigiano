import sklearn
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import torch


def predict_pergene(data, params, posterior_stats, group):
    performance = {}
    num_indivs = data.G[group].shape[0]
    w_g = torch.tensor(posterior_stats['w_g']['mean'], dtype = torch.float32)
    rho_g = torch.tensor(posterior_stats['rho']['mean'], dtype = torch.float32)
    Z_norm = torch.tensor(posterior_stats["Z"]['mean'], dtype = torch.float32)
    alpha = torch.tensor(posterior_stats['alpha']['mean'], dtype = torch.float32)
    lambda_ = ((data.Z.T * data.maf_weights).T.matmul(data.tau)) * w_g
    beta = rho_g * lambda_ + (1-rho_g) * lambda_ * Z_norm
    Gbeta = (data.G[group]).matmul(beta) 
    preds = torch.sigmoid(torch.matmul(data.X[group], alpha).reshape(-1,1) + Gbeta.reshape(-1,1)).view(num_indivs) 
    fpr, tpr, thresholds = roc_curve(np.array(data.AD_status[group]), preds.detach().numpy())
    performance['AUC'] = auc(fpr, tpr)
    performance['ACC'] = ((preds > 0.5).float().detach().numpy() == np.array(data.AD_status[group])).sum() / len(preds)
    return performance
    

def predict_joint(data, params, posterior_stats, group):
    num_indivs = data.G[group].shape[0]
    performance = {}
    performance['AUC'] = {}
    performance['ACC'] = {}
    tau = torch.tensor(posterior_stats['tau']['mean'], dtype = torch.float32)
    w_g = torch.tensor(posterior_stats['w_g']['mean'], dtype = torch.float32)
    rho_g = torch.tensor(posterior_stats['rho_g']['mean'], dtype = torch.float32)
    for gene in range(data.num_genes):
        Z_norm = torch.tensor(posterior_stats[f"Z_{data.genes[gene]}"]['mean'], dtype = torch.float32)
        if params['alpha_gene']:
            alpha = torch.tensor(posterior_stats[f'alpha_{data.genes[gene]}']['mean'], dtype = torch.float32)
        else:
            alpha = torch.tensor(posterior_stats['alpha']['mean'], dtype = torch.float32)
        lambda_ = ((data.Zs[data.gene_indices==gene].T * data.maf_weights[data.gene_indices == gene]).T.matmul(tau)) * w_g[gene] 
        beta = rho_g[gene]  * lambda_ + (1-rho_g[gene]) * lambda_ * Z_norm
        Gbeta = data.G[group][:,data.gene_indices==gene].matmul(beta)
        preds = torch.sigmoid(torch.matmul(data.X[group], alpha).reshape(-1,1) + Gbeta.reshape(-1,1)).view(num_indivs)
        if params['simulate']:
            fpr, tpr, thresholds = roc_curve(np.array(data.AD_status[group][data.genes[gene]]), preds.detach().numpy())
            aucs[data.genes[gene]] = auc(fpr, tpr)
            accs[data.genes[gene]] = ((preds > 0.5).float().detach().numpy() == np.array(data.AD_status[group][data.genes[gene]])).sum() / len(preds)
        else:
            fpr, tpr, thresholds = roc_curve(np.array(data.AD_status[group]), preds.detach().numpy())
            performance['AUC'][data.genes[gene]] = auc(fpr, tpr)
            performance['ACC'][data.genes[gene]] = ((preds > 0.5).float().detach().numpy() == np.array(data.AD_status[group])).sum() / len(preds)
    return performance
