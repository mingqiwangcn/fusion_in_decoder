import torch
import torch.nn as nn

class EnsembleRetrModel(nn.Module):
    def __init__(self, fabric_model):
        self.fabric_model = fabric_model
        super(EnsembleRetrModel, self).__init__()
    
    def forward(self, batch_data, batch_examples, fusion_scores, fusion_states):
        opt_info = {}
        fabric_scores = self.fabric_model(batch_data, batch_examples, opt=opt_info)
        bsz = len(batch_data)
        batch_retr_scores = []
        for idx in range(bsz):
            fabric_passage_scores = fabric_scores[idx]
            fusion_passage_scores = fusion_scores[idx]
            score_weights = self.compute_score_weights(fabric_states, fusion_states)
            retr_scores = None
            batch_retr_scores.append(retr_scores)
        return batch_retr_scores
   
    def get_fabric_states(self, opt_info):
        return
    
    def compute_score_weights(self, fabric_states, fusion_states):
        return 
