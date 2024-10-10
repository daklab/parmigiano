import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta, AutoMultivariateNormal, AutoNormal
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.infer import Predictive
from torch.distributions import constraints
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os, sys
from dataclasses import dataclass, field
import yaml
import sklearn
from sklearn import metrics 
import json
import pickle
import torch.nn.functional as F
from sklearn.decomposition import NMF
import utils
import data_class
from dataclasses import fields
import load_data
import models
from sklearn.linear_model import LogisticRegression


##### INPUT ARGUMENTS ####
params_file = sys.argv[1]
with open(params_file, 'r') as stream:
    params = yaml.safe_load(stream)  
CHRO_NB = int(sys.argv[2])
params['genes'] = CHRO_NB
##########################


def fit(data, params):
    if params['simulate']:
        data = getattr(models, params['model'])().forward(data, params, True)
    model = getattr(models, params['model'])()
    guide = AutoGuideList(model)
    to_optimize = ['rho']
    guide.add(AutoNormal(poutine.block(model, hide = to_optimize)))
    guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))
    adam = pyro.optim.Adam({"lr": params['lr']})
    svi = SVI(model, guide = guide, optim = adam, loss=Trace_ELBO()) 
    pyro.clear_param_store()
    for j in range(params['epochs']): 
        loss = svi.step(data, params) 
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=10) 
    samples = predictive(data, params)
    posterior_stats = {k:{'mean':np.array(torch.mean(v, 0)[0]),
                         'std': np.array(torch.std(v,0)[0])} for k,v in samples.items() if "obs" not in k}
    results = {}
    for variable in posterior_stats:
        if variable in ['w_g', 'w2g', 'rho']:
            results[variable] = float(posterior_stats[variable]['mean'])
            results[variable + "_std"] = float(posterior_stats[variable]['std'])
    if params['simulate']:
        fields_list = list(data.__dict__.keys())
        for var in fields_list:
            if var in ['wg', 'rho', 'w2g']:
                results['true_' + var] = float(getattr(data, var))
            if var == 'Z_norm':
                results['Z_R'] = np.corrcoef(data.Z_norm, posterior_stats['Z']['mean'])[0,1]
    return results

def main():
    if params['simulate']: 
        params['output_path'] = os.path.join(params['output_path'], 'simulation', params['output'])
    else:
        params['output_path'] = os.path.join(params['output_path'], 'true', params['output'])
    if not os.path.exists(params['output_path']):
        os.mkdir(params['output_path'])
    params['test_prop'] = 0.001
    if (params['enformer_preds']) & (params['cell'] != "coding"):
        enformer = pd.read_csv(params['enformer_path'], sep = "\t")
        delta_columns = [col for col in enformer.columns if 'delta' in col]
        enformer[delta_columns] = enformer[delta_columns].abs()
    tau, _, _ = load_data.load_model(params['jointly_trained_model']) # pre-trained parameters
    X, Y, genes, skip = load_data.read_data(params)
    stats = {}
    i = 0
    for GENE in tqdm(genes[CHRO_NB]):
        try:
            Gs, Zs = load_data.load_gene(GENE, CHRO_NB, params)
        except: 
            continue
        if params['enformer_preds']: 
            Zs = Zs.merge(enformer, left_on = 'variant_id', right_on = 'SNP').drop(['CHR','SNP','BP','A1','A2'], axis = 1)
        with open('/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_matrices_maf/min_max_scaling.pkl', 'rb') as file:
            minmax = pickle.load(file)
        Zs = Zs.set_index(['variant_id', 'TargetGene']) 
        for column in Zs.columns:
            if column in minmax[params['cell']]['min'] and column in minmax[params['cell']]['max']:
                min_val = minmax[params['cell']]['min'][column]
                max_val = minmax[params['cell']]['max'][column]
                Zs[column] = Zs[column].clip(lower=min_val, upper=max_val)
                Zs[column] = (Zs[column] - min_val) / (max_val - min_val)
        Zs['intercept'] = 1 
        Zs = Zs.fillna(0) 
        data = data_class.PerGeneAD.from_pandas(Gs, Zs, X, Y, params)
        data.tau = tau
        stats[GENE] = fit(data, params)
        if i % 30 == 0:
            df = pd.DataFrame(stats).T
            df.to_csv(os.path.join(params['output_path'], 'chr' + str(CHRO_NB) + '.csv'))
        i += 1
    df = pd.DataFrame(stats).T
    df.to_csv(os.path.join(params['output_path'], 'chr' + str(CHRO_NB) + '.csv'))
    return


if __name__ == "__main__":
    main()