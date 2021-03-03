import torch
import math
import numpy as np


DEFAULT_MU = torch.ones([4]) * 4
DEFAULT_SIGMA = torch.ones([4, 4]) * .1 + torch.eye(4)
DEFAULT_SIGMA_INV = torch.inverse(DEFAULT_SIGMA)


scale = 4
centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
centers = [(scale * x, scale * y) for x, y in centers]
DEFAULT_CENTERS = torch.tensor(centers)

# DEFAULT_CENTERS = torch.stack([torch.ones([4]) * i for i in range(8)])

def gaussian_score(z, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA):
    mu = mu.to(z)
    sigma = sigma.to(z)

    assert mu.shape[-1] == sigma.shape[-1]

    sigma_inv = torch.inverse(sigma).to(z).unsqueeze(0).expand(z.shape[0], -1, -1)
    score = torch.matmul(sigma_inv, -z.unsqueeze(2) + mu.unsqueeze(1))
    return score.squeeze()



class IsometricGaussianMollifier(object):
    def __init__(self, sigma_square=5e-2):
        self.sigma_square = sigma_square

    def score(self, centers, particles):
        '''
        This function computes the score of the Gaussian mixture at particles.
        The centers of the Gaussian mixture is given in centers. The variance of the Gaussians are self.sigma_square
        Args:
            centers: centers of the Gaussian mixture
            particles: position of the particles

        Returns: score of the Gaussian mixture at particles

        '''
        z = particles.unsqueeze(dim=1).expand(-1, centers.shape[0], -1) - centers
        b = torch.sum(z.pow(2), dim=2) / 2 / self.sigma_square
        a = torch.exp(-b) + 1e-12
        normalized_a = (a / a.sum(dim=1, keepdim=True)).unsqueeze(dim=2)

        # c = -z+centers
        score = torch.sum(-z * normalized_a/self.sigma_square, dim=1)

        # print(torch.norm(score - score_2))
        return score


def gaussian_mixture_score(z, centers):

    centers = torch.tensor(centers).to(z)

    z = z.unsqueeze(dim=1).expand(-1, centers.shape[0], -1) - centers
    a = torch.sum(z.pow(2), dim=2)/2.
    a = torch.exp(-a)
    normalized_a = (a/a.sum(dim=1, keepdim=True)).unsqueeze(dim=2)
    # c = -z+centers
    score = torch.sum(-z*normalized_a, dim=1)

    # print(torch.norm(score - score_2))
    return score


def gaussian_mixture_logprob(z):
    # centers = torch.tensor([(4, 4), (4, 3), (4, 2)]).to(z)
    # centers = torch.tensor([(4, 4)]).to(z)

    scale = 4.
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / math.sqrt(2), 1. / math.sqrt(2)),
               (1. / math.sqrt(2), -1. / math.sqrt(2)), (-1. / math.sqrt(2),
                                                         1. / math.sqrt(2)), (-1. / math.sqrt(2), -1. / math.sqrt(2))]
    centers = [(scale * x, scale * y) for x, y in centers]
    centers = torch.tensor(centers).to(z)

    # centers = torch.tensor([(4, 4)]).to(z)


    # batch_size = z.shape[0]
    # z_logprob = []
    # normalizing_factor = torch.tensor(1./(2.*math.pi))
    # for i in range(batch_size):
    #     a = torch.sum((z[i] - centers).pow(2), dim=1)/2.
    #     z_logprob.append(torch.log(torch.mean(torch.exp(-a)*normalizing_factor).unsqueeze(dim=0)))
    #
    # z_logprob = torch.cat(z_logprob, dim=0)

    z = z.unsqueeze(dim=1).expand(-1, centers.shape[0], -1) - centers
    a = torch.sum(z.pow(2), dim=2) / 2.
    a = torch.exp(-a)
    a = torch.log(torch.mean(a, dim=1))

    z_logprob = a - torch.log(torch.tensor(2.*math.pi))
    # print(torch.norm(z_logprob2 - z_logprob)/torch.norm(z_logprob))

    return z_logprob

def gaussian_logprob(z, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA):
    mu = mu.to(z)
    sigma = sigma.to(z)

    assert mu.shape[0] == sigma.shape[0]

    #TODO: implement this!
    assert(False)
    mu = torch.tensor([(4., 4.)]).to(z)
    sigma_half = torch.tensor([(2., 1.), (1., 2.)]).to(z)
    sigma_half_inv = torch.inverse(sigma_half).to(z)
    sigma = sigma_half.matmul(sigma_half).to(z)
    sigma_inv = torch.inverse(sigma).to(z)
    sigma_det = torch.det(sigma)

    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - sigma_inv.matmul(z-mu) / 2.
