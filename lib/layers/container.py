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

    def forward(self, x, logpz, score, wgf_reg):
        for i in range(len(self.chain)):
            x, logpz, score, wgf_reg = self.chain[i](x, logpz, score, wgf_reg)
            return x, logpz, score, wgf_reg