import torch.nn as nn


class SequentialFlowTest(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlowTest, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, y, score, diff_score, inds=None):
        if inds is None:
            inds = range(len(self.chain))

        for i in inds:
            x, y, score, diff_score = self.chain[i](x, y, score, diff_score)
        return x, y, score, diff_score


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpz, score, wgf_reg, mu_0=None, sigma_half_0=None, score_error_0=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpz is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpz, score, wgf_reg, mu, sigma_half, score_error = self.chain[i](x, logpz, score, wgf_reg, mu_0, sigma_half_0, score_error_0, reverse=reverse)
            return x, logpz, score, wgf_reg, mu, sigma_half, score_error