import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionGeneralRetrLoss(nn.Module):
    def __init__(self):
        super(FusionGeneralRetrLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch_score, batch_answers):
        batch_loss = .0
        batch_num = 0
        
        for b_idx, item_scores in enumerate(batch_score):
            answer_lst = batch_answers[b_idx]
            assert(len(item_scores) == len(answer_lst))
            
            #pos_idxes, neg_idxes = self.get_pos_neg_idxes(answer_lst)
            #if (len(pos_idxes) == 0) or (len(neg_idxes) == 0):
            #    continue

            labels = [(1 if a['em'] >= 1 else 0) for a in answer_lst]
            labels = torch.tensor(labels).float().to(item_scores.device)
            item_loss = self.loss_fn(item_scores, labels)
             
            batch_loss += item_loss
            batch_num += 1

        if batch_num > 0: 
            loss = batch_loss / batch_num
        else:
            loss = None
        return loss
    
    def get_pos_neg_idxes(self, answer_lst):
        pos_idxes = []
        neg_idxes = []
        for idx, answer in enumerate(answer_lst):
            em_score = answer['em']
            if em_score >= 1:
                pos_idxes.append(idx)
            else:
                neg_idxes.append(idx) 
        return pos_idxes, neg_idxes 


