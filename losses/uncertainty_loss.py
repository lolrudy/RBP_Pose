import numpy as np
import torch

def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, balance_weight, reduction='mean', sum_last_dim=False):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    if sum_last_dim:
        loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target).sum(-1) + balance_weight * 0.5 * log_variance
    else:
        loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + balance_weight * 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()



if __name__ == '__main__':
    pass
