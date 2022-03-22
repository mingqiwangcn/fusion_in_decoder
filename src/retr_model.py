import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionRetrModel(nn.Module):
    def __init__(self):
        super(FusionRetrModel, self).__init__()
        D = 768
        
        self.feature_fnt = nn.Sequential(
                        nn.Linear(D * 3, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
        )
         
    def std_norm(self, scores):
        mean_score = scores.mean()
        std_score = scores.std()
        ret_scores = (scores - mean_score) / (std_score + 1e-5)
        return ret_scores

    def forward(self, batch_data, fusion_scores, fusion_states, passage_masks):
        answer_states = fusion_states['answer_states'] # (bsz, n_layers, n_tokens, emb_size)
        #answer_states = answer_states[:, -1:, :, :]
        query_passage_states = fusion_states['query_passage_states'] 
        #query_passage_states = query_passage_states[:, -1:, :, :]
        bsz, n_layers, n_tokens, emb_size = query_passage_states.size()
        
        input_states = [answer_states, query_passage_states, answer_states * query_passage_states]
        input_states = torch.cat(input_states, dim=-1)
       
        batch_scores = self.feature_fnt(input_states).squeeze(-1)
       
        batch_passage_scores = []
        for idx in range(len(batch_data)):
            n_passages = len(batch_data[idx]['passages'])
            item_masks = passage_masks[idx].expand(n_layers, -1, -1)
            item_scores = batch_scores[idx].view(n_layers, n_passages, -1)
            item_masked_scores = item_scores * item_masks
            item_adapt_scores = item_masked_scores.sum(dim=[0,2])
            item_fusion_scores = fusion_scores[idx]
            passage_scores = self.std_norm(item_adapt_scores) # + self.std_norm(item_fusion_scores) 
            batch_passage_scores.append(passage_scores)
                 
        return batch_passage_scores 
         
