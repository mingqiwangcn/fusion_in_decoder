import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionRetrModel(nn.Module):
    def __init__(self):
        super(FusionRetrModel, self).__init__()
        D = 768
        self.fusion_fnt = nn.Sequential(
                        nn.Linear(D * 3, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
        )
         
    def forward(self, batch_data, fusion_scores, fusion_states):
        fusion_scores_redo = self.recompute_fusion_score(batch_data, fusion_scores, fusion_states)
        return fusion_scores_redo
  
    def recompute_fusion_score(self, batch_data, fusion_scores, fusion_states):
        answer_states = fusion_states['answer_states']
        bsz, n_layers, _, emb_size = answer_states.size()
        query_passage_states = fusion_states['query_passage_states'] 
        _, _, n_tokens, _ = query_passage_states.size()
        
        answer_states = answer_states.expand(bsz, n_layers, n_tokens, emb_size)
        input_states = [answer_states, query_passage_states, answer_states * query_passage_states]
        input_states = torch.cat(input_states, dim=-1)
        batch_scores = self.fusion_fnt(input_states).squeeze(-1)
        
        batch_passage_scores = []
        for idx in range(len(batch_data)):
            n_passages = len(batch_data[idx]['passages'])
            item_scores = batch_scores[idx].view(n_layers, n_passages, -1)
            passage_scores = item_scores.sum(dim=[0,2]) # + fusion_scores[idx]
            batch_passage_scores.append(passage_scores)
                 
        return batch_passage_scores 
         
