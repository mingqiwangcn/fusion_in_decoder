from src.general_retr_loss import FusionGeneralRetrLoss
import torch
import torch.nn as nn

class FusionGeneralRetrBNNLoss(nn.Module):
    def __init__(self):
        super(FusionGeneralRetrBNNLoss, self).__init__()
        self.mle_loss_fnt = FusionGeneralRetrLoss()
         
    def forward(self, batch_score, batch_answers, opts=None):
        mle_loss = self.mle_loss_fnt(batch_score, batch_answers, opts=opts)
        loss = (opts['log_variational_posterior'] - opts['log_prior']) / len(batch_score) + mle_loss
        return loss

