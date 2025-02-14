## Use score-based testing to get gene-level significance from parmigiano, using global annotation weights tau
import utils
import data_class
import load_data
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import os, sys
from dataclasses import dataclass, field
import yaml
import pickle
import time

from numpy.linalg import svd, solve
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats.mvn import mvnun
from statsmodels.sandbox.distributions.extras import mvnormcdf

##### INPUT ARGUMENTS ####
params_file = sys.argv[1]
with open(params_file, 'r') as stream:
    params = yaml.safe_load(stream)  
CHRO_NB = int(sys.argv[2])
params['cell'] = sys.argv[3]
params['genes'] = CHRO_NB

d = {'peripheralPU1nuclei': {'jointly_trained_model': '/gpfs/commons/home/adas/parmigiano/outputs/true_full/microglia_01',
                             'enformer_path': '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_matrices_maf/ADSP_rare_variants_enformer_delta_scores_annotations_microglia.tsv.gz'},
    'NEUN': {'jointly_trained_model': "/gpfs/commons/home/adas/parmigiano/outputs/true_full/neuron_01",
                             'enformer_path': '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_matrices_maf/ADSP_rare_variants_enformer_delta_scores_annotations_neuron.tsv.gz'},
    'OLIG2': {'jointly_trained_model': "/gpfs/commons/home/adas/parmigiano/outputs/true_full/oligodendrocyte_01",
                             'enformer_path': '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_matrices_maf/ADSP_rare_variants_enformer_delta_scores_annotations_oligodendrocyte.tsv.gz'},
    'LHX2': {'jointly_trained_model': "/gpfs/commons/home/adas/parmigiano/outputs/true_full/astrocyte_01",
                             'enformer_path': '/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_matrices_maf/ADSP_rare_variants_enformer_delta_scores_annotations_astrocyte.tsv.gz'},
    'coding': {'jointly_trained_model': '/gpfs/commons/home/adas/parmigiano/outputs/true_full/coding_01',
                             'enformer_path': False}}

params['jointly_trained_model'] = d[params['cell']]['jointly_trained_model']
params['enformer_path'] = d[params['cell']]['enformer_path']

##########################



def null_model(data):
    '''
    Null Model: Y ~ X
    '''
    prelim_result = sm.GLM(np.array(data.AD_status['train']), 
                           np.array(data.X['train']),
                           family=sm.families.Binomial()).fit()
    return prelim_result


def get_scores(prelim_result, data):
    '''
    mu_hat = predicted AD status from covariate-only null model [N x 1]
    y_res = y - mu_hat [N x 1]
    score = G * y_res: variant scores, larger --> variant is more likely to contribute to residual [P x 1]
    '''
    mu_hat = prelim_result.predict(np.array(data.X['train']))
    y_res = data.AD_status['train'] - torch.tensor(mu_hat, dtype = torch.float32)
    score = data.G['train'].T.matmul(y_res)
    return score


def get_resampled_score(prelim_result, data, B = 10000):
    '''
    Generate resampled variant scores. Used for approximating distribution of MinP and Fisher's test statistics.
    
    S = variant score = G * y_res
    Sigma = covariance(S) = G^T (V - VX (X^TVX)^-1 X^TV) G
        - V = mu_hat (1 - mu_hat) for binary traits
    Sample B x 1 score vectors: ~MVN (0, Sigma)
    '''
    n, p = data.G['train'].shape
    X = data.X['train']
    G = data.G['train']
    mu_hat = prelim_result.predict(np.array(X))
    v = torch.tensor(mu_hat * (1-mu_hat), dtype = torch.float32)
    vG = v[:, None] * G  # Element-wise multiplication with broadcasting
    vX0 = v[:, None] * X
    Sigma = vG.T @ G - (vG.T @ X) @ solve(vX0.T @ X, vX0.T @ G) # This is not always positive semi definite -- throws errors in Pyro ~MVN sampling
    mean = np.zeros(p)  # Mean vector of zeros
    score_re_var = np.random.multivariate_normal(mean, Sigma, size=B) # Resulting resampled scores
    return score_re_var


def get_Q_statistics(data, scores, score_re_var):
    '''
    Calculate burden and dispersion Q-statistics
    '''
    #Z_w = data.maf_weights.reshape(-1,1) * data.Z 
    Z_w = ((data.Z.T * data.maf_weights).T.matmul(data.tau)).reshape(-1,1)
    score_re_var = torch.tensor(score_re_var, dtype = torch.float32)
    variant_scores = torch.cat((scores.reshape(1,-1), score_re_var))
    Q_disp = (variant_scores**2).matmul(Z_w**2)
    Q_burd = (variant_scores.matmul(Z_w))**2
    return Q_disp, Q_burd

def get_Q_pvals(Q, re_Q):
    """
    Calculate p-values for test statistics based on resampled test statistics.
    
    Parameters:
    Q (numpy.ndarray): A (A x q) matrix of test statistics.
    re_Q (numpy.ndarray): A (B x q) matrix of resampled test statistics.

    Returns:
    numpy.ndarray: A (A x q) matrix of p-values.
    """
    re_mean = np.mean(re_Q, axis=0)  # Mean along rows for each column
    re_variance = np.var(re_Q, axis=0, ddof=1)  # Variance along rows for each column
    re_kurtosis = np.mean((re_Q - re_mean) ** 4, axis=0) / (re_variance ** 2) - 3
    re_df = np.where(re_kurtosis > 0, 12 / re_kurtosis, 100000)
    Q_adjusted = (Q - re_mean) * np.sqrt(2 * re_df) / np.sqrt(re_variance) + re_df
    re_p = chi2.sf(Q_adjusted, re_df)  # Survival function for chi-squared distribution
    return re_p

def rho_pvals(data, Q_burd, Q_disp, B = 10000):
    """
    Calculate p-values for test statistics across multiple rho values.
    
    Parameters:
    - data: Dataset containing required matrices and variables.
    - Q1, Q2: Test statistic matrices for two components.
    - re_Q1, re_Q2: Resampled test statistic matrices for two components.
    - rho_class: List or array of rho values to evaluate.
    - get_p_function: Function to calculate p-values (equivalent to `Get.p` in R).

    Returns:
    - re_p: Resampled p-values (all rows except the first).
    - temp_p: Observed p-values (first row).
    """
    rho_class = params['rho'] #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    q = Q_burd.shape[1]  # Number of test statistics -- for parmigiano this will be ONE
    all_p = np.zeros((B + 1, len(rho_class), q))
    Q1 = Q_disp[0]
    re_Q1 = Q_disp[1:]
    Q2 = Q_burd[0]
    re_Q1 = Q_burd[1:]
    for i, rho in enumerate(rho_class):
        Q_rho = (1 - rho) * Q_disp + rho * Q_burd
        # Calculate p-values using the provided function
        all_p[:, i, :] = get_Q_pvals(np.array(Q_rho), np.array(Q_rho[1:]))  
    # Extract resampled and observed p-values
    re_p = all_p[1:, :, :]  # All rows except the first, resampled vals
    temp_p = all_p[0, :, :]  # Only the first row, observed vals
    return re_p, temp_p

def FCombine_p(p, re_p):
    """
    Combine multiple p-values using Fisher's method.

    Parameters:
    - p: numpy array, observed p-values.
    - re_p: numpy array, resampled p-values (B x b x c).

    Returns:
    - p_combined: float, combined p-value.
    """
    Fisher_stat = -2 * np.sum(np.log(p))
    re_Fisher = -2 * np.nansum(np.log(re_p), axis=1)
    re_Fisher[re_Fisher == np.inf] = np.nan  # Replace infinities with NaN
    Fisher_mean = np.nanmean(re_Fisher)
    Fisher_variance = np.nanvar(re_Fisher, ddof=1)
    Fisher_kurtosis = np.nanmean((re_Fisher - Fisher_mean)**4) / (Fisher_variance**2) - 3
    if Fisher_kurtosis > 0:
        df = 12 / Fisher_kurtosis
    else:
        df = 100000
    adjusted_stat = (Fisher_stat - Fisher_mean) * np.sqrt(2 * df) / np.sqrt(Fisher_variance) + df
    p_combined = chi2.sf(adjusted_stat, df)
    return p_combined


def mcombine_p(p, re_p):
    """
    Combine multiple p-values for a MinP test.
    
    Parameters:
    p (array-like): Array of p-values.
    re_p (array-like): A b x c matrix of resampled p-values.
    
    Returns:
    float: Combined p-value.
    """
    min_p_stat = np.min(p)
    min_p_rho = params['rho'][np.argmin(p)]
    re_normal = norm.ppf(re_p)
    re_normal[np.isinf(re_normal)] = np.nan
    D = np.array(pd.DataFrame(re_normal).corr())#np.corrcoef(re_normal, rowvar=False)
    
    # Compute the multivariate normal cumulative probability
    try:
        qnorm_min = norm.ppf(min_p_stat)
        lower_bound = np.full(len(p), qnorm_min)
        mean = np.zeros(len(p))
        p_combined = 1 - mvnormcdf(upper=np.inf, mu=mean, cov=D, lower=lower_bound, maxpts = 5000*len(p))
    except Exception as e:
        print(f"Error in multivariate normal computation: {e}")
        p_combined = np.nan
    
    # Handle extreme p-values
    if p_combined == 0:
        print("Warning: Extreme p-value < 1e-15 is not supported by MinP method. "
              "P-value is recorded as 1e-15.")
        p_combined = 1e-15
    return p_combined, min_p_rho



def gene_P(data, prelim_result):
    scores = get_scores(prelim_result, data)
    p = {}
    for iteration in range(params['iterations']): # run 10x 
        score_re_var = get_resampled_score(prelim_result, data, B = params['B'])
        Q_disp, Q_burd = get_Q_statistics(data, scores, score_re_var)
        re_p, temp_p = rho_pvals(data, Q_burd, Q_disp, B = params['B'])
        minp, minrho = mcombine_p(temp_p.reshape(-1,), re_p[:,:,0])
        p [iteration] = {'MinP': minp, 'Rho': minrho}
    return p



def main():
    if params['simulate']: 
        params['output_path'] = os.path.join(params['output_path'], 'simulation', params['output'])
    elif params['permute_Y']: 
        params['output_path'] = os.path.join(params['output_path'], 'permutation', params['output'])
    else:
        params['output_path'] = os.path.join(params['output_path'], 'true', params['output'])
    if not os.path.exists(params['output_path']):
        os.makedirs(params['output_path'], exist_ok = True)
    params['output_path'] = os.path.join(params['output_path'], params['cell'])
    if not os.path.exists(params['output_path']):
        os.makedirs(params['output_path'], exist_ok = True)

    params['test_prop'] = 0
    if (params['enformer_preds']) & (params['cell'] != "coding"):
        enformer = pd.read_csv(params['enformer_path'], sep = "\t")
        delta_columns = [col for col in enformer.columns if 'delta' in col]
        enformer[delta_columns] = enformer[delta_columns].abs()
    if (params['cell'] == "coding") & (params['lof_preds'] == True): 
        lof, missense = load_data.load_lof(params, CHRO_NB)
    tau, _, _ = load_data.load_model(params['jointly_trained_model']) # pre-trained parameters
    X, Y, genes, skip = load_data.read_data(params)
    results = {}
    timing = {}
    i = 0
    for GENE in tqdm(genes[CHRO_NB]):
        try:
            Gs, Zs = load_data.load_gene(GENE, CHRO_NB, params)
        except: 
            print("Issue with ", GENE)
            continue
        if (params['enformer_preds']) & (params['cell'] != "coding"): 
            Zs = Zs.merge(enformer, left_on = 'variant_id', right_on = 'SNP').drop(['CHR','SNP','BP','A1','A2'], axis = 1)
        with open('/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/Alzheimer-RV/data/gene_matrices_maf/min_max_scaling.pkl', 'rb') as file:
            minmax = pickle.load(file)
        if (params['cell'] == "coding") & (params['lof_preds'] == True):
            anno_str = Zs['variant_id'].str.split("_").str[0].str.split("chr").str[1]
            Zs['lof'] = np.where(anno_str.isin(list(lof['level_1'])), 1, 0)
            Zs['missense'] = np.where(anno_str.isin(list(missense[1])), 1, 0)
        Zs = Zs.set_index(['variant_id', 'TargetGene']) 
        for column in Zs.columns:
            if column in minmax[params['cell']]['min'] and column in minmax[params['cell']]['max']:
                min_val = minmax[params['cell']]['min'][column]
                max_val = minmax[params['cell']]['max'][column]
                Zs[column] = Zs[column].clip(lower=min_val, upper=max_val)
                Zs[column] = (Zs[column] - min_val) / (max_val - min_val)
        Zs['intercept'] = 1 
        Zs = Zs.fillna(0) 
        if params['MAF'] < 0.05:
            Zs = Zs.iloc[np.where(Gs.mean(0)/2 < params['MAF'])[0]]
            Gs = Gs[Gs.columns[np.where(Gs.mean(0)/2 < params['MAF'])[0]]]
        data = data_class.PerGeneAD.from_pandas(Gs, Zs, X, Y, params)
        if params['permute_Y']:
            indices = torch.randperm(data.AD_status['train'].size(0))
            data.AD_status['train'] = data.AD_status['train'][indices]
        data.wg_prior = 0.0
        data.tau = tau
        data.X['train'] = torch.linalg.svd(data.X['train'][:,:-1], full_matrices = False).U # SVD to account for multicollinearity
        ones_column = torch.ones(data.X['train'].size(0), 1)
        data.X['train'] = torch.cat([data.X['train'] , ones_column], dim=1)
        
        if params['simulate']:
            data = getattr(models, params['model'])().forward(data, params, True)
        if (i == 0):
            prelim_result = null_model(data)
        if (params['simulate'] == True) | (params['permute_Y'] == True):
            prelim_result = null_model(data) # Need to perform null model each iteration if so.
        start = time.time()
        results[GENE] = gene_P(data, prelim_result)  
        result_time = time.time() - start
        timing[GENE] = {"Time": result_time, "Variants": data.G['train'].shape[1]}
        if i % 30 == 0:
            path = os.path.join(params['output_path'], 'chr' + str(CHRO_NB) + '.pkl')
            with open(path, 'wb') as f:
                pickle.dump(results, f)
            path = os.path.join(params['output_path'], 'timing_chr' + str(CHRO_NB) + '.pkl')
            with open(path, 'wb') as f:
                pickle.dump(timing, f)
        i += 1
    path = os.path.join(params['output_path'], 'chr' + str(CHRO_NB) + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    path = os.path.join(params['output_path'], 'timing_chr' + str(CHRO_NB) + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(timing, f)
    return

    
if __name__ == "__main__":
    main()
