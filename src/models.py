import data_class
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


class gruyereSKAT(PyroModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, data, params):
        num_indivs = data.G['train'].shape[0]
        w_g = pyro.sample("w_g", dist.Normal(0,1))
        alpha = pyro.sample('alpha', dist.Normal(0,1).expand([data.num_cov]).to_event(1))   
        lambda_ = ((data.Z.T * data.maf_weights).T.matmul(data.tau)) * w_g
        Z_norm = pyro.sample("Z", dist.Normal(0,1).expand([len(data.maf_weights)]).to_event(1))
        beta = lambda_ * Z_norm
        if params['simulate']:
            data.w_g = w_g
            data.alpha = alpha
            data.Z_norm = Z_norm        
        Gbeta = (data.G['train']).matmul(beta) 
        mean = torch.sigmoid(torch.matmul(data.X['train'], alpha).reshape(-1,1) + Gbeta.reshape(-1,1)).view(num_indivs) 
        with pyro.plate('data', num_indivs):
            if params['simulate']:
                data.AD_status['train'] = pyro.sample("obs", dist.Bernoulli(mean))
                return data
            else:
                obs = pyro.sample('obs', dist.Bernoulli(mean), obs = data.AD_status['train'])
        return 
    
class gruyereBurden(PyroModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, data, params):
        num_indivs = data.G['train'].shape[0]
        w_g = pyro.sample("w_g", dist.Normal(0,1))
        alpha = pyro.sample('alpha', dist.Normal(0,1).expand([data.num_cov]).to_event(1))   
        lambda_ = ((data.Z.T * data.maf_weights).T.matmul(data.tau)) * w_g
        if params['simulate']:
            data.w_g = w_g
            data.alpha = alpha
        Gbeta = (data.G['train']).matmul(lambda_) 
        mean = torch.sigmoid(torch.matmul(data.X['train'], alpha).reshape(-1,1) + Gbeta.reshape(-1,1)).view(num_indivs) 
        with pyro.plate('data', num_indivs):
            if params['simulate']:
                data.AD_status['train'] = pyro.sample("obs", dist.Bernoulli(mean))
                return data
            else:
                obs = pyro.sample('obs', dist.Bernoulli(mean), obs = data.AD_status['train'])
        return 
    
class gruyereO(PyroModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, data, params):
        num_indivs = data.G['train'].shape[0]
        rho = pyro.sample("rho", dist.Beta(0.5, 0.5))
        w_g = pyro.sample("w_g", dist.Normal(0,1))
        alpha = pyro.sample('alpha', dist.Normal(0,1).expand([data.num_cov]).to_event(1))   
        lambda_ = ((data.Z.T * data.maf_weights).T.matmul(data.tau)) * w_g
        Z_norm = pyro.sample("Z", dist.Normal(0,1).expand([len(data.maf_weights)]).to_event(1))
        beta = rho * lambda_ + (1-rho) * lambda_ * Z_norm
        if params['simulate']:
            data.w_g = w_g
            data.alpha = alpha
            data.Z_norm = Z_norm  
            data.rho = rho
        Gbeta = (data.G['train']).matmul(beta) 
        mean = torch.sigmoid(torch.matmul(data.X['train'], alpha).reshape(-1,1) + Gbeta.reshape(-1,1)).view(num_indivs) 
        with pyro.plate('data', num_indivs):
            if params['simulate']:
                data.AD_status['train'] = pyro.sample("obs", dist.Bernoulli(mean))
                return data
            else:
                obs = pyro.sample('obs', dist.Bernoulli(mean), obs = data.AD_status['train'])
        return 