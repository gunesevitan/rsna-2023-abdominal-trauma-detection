from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class SampleWeightedBCEWithLogitsLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):

        super(SampleWeightedBCEWithLogitsLoss, self).__init__(weight=weight, reduction=reduction)

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weights):

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=self.weight)
        loss = loss * sample_weights

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SampleWeightedBCELoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):

        super(SampleWeightedBCELoss, self).__init__(weight=weight, reduction=reduction)

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weights):

        loss = F.binary_cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        loss = loss * sample_weights

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SampleWeightedCrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):

        super(SampleWeightedCrossEntropyLoss, self).__init__(weight=weight, reduction=reduction)

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weights):

        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        loss = loss * sample_weights

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
