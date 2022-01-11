import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleRetrModel(nn.Module):
    def __init__(self, fabric_model):
        super(EnsembleRetrModel, self).__init__()
        self.fabric_model = fabric_model
        
        D = 768
        self.fusion_fnt = nn.Sequential(
                        nn.Linear(D * 3, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
        )
        
        self.expert_fnt = nn.Sequential(
                        nn.Linear(D * 7, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 3)
        )
         
    def forward(self, batch_data, batch_examples, fusion_scores, fusion_states):
        opt_info = {}
        fabric_scores = self.fabric_model(batch_data, batch_examples, opts=opt_info)
        bsz = len(batch_data)
        batch_retr_scores = []
        fusion_scores_redo, fusion_input_states = self.recompute_fusion_score(fusion_states, batch_data)
        
        fabric_input_states = self.get_fabric_input_states(opt_info, batch_data)

        for idx in range(bsz):
            fabric_passage_scores = torch.cat(fabric_scores[idx])
            fusion_passage_scores = fusion_scores[idx]
            fusion_passage_scores_redo = fusion_scores_redo[idx]
            
            fabric_item_input_states = fabric_input_states[idx]
            fusion_item_input_states = fusion_input_states[idx]
            
            n_layers, n_tokens, emb_size = fusion_item_input_states.size()
            n_passages = len(batch_data[idx]['passages'])
            fusion_item_input_states = fusion_item_input_states.view(n_layers, n_passages, -1, emb_size)
            fusion_item_input_states = fusion_item_input_states.sum(dim=[0,2]) 

            #(n_passages, 3)
            score_weights = self.compute_score_weights(fabric_item_input_states, fusion_item_input_states)
            
            expert_scores = torch.cat([
                fabric_passage_scores.view(n_passages, -1),
                fusion_passage_scores.view(n_passages, -1),
                fusion_passage_scores_redo.view(n_passages, -1)
            ], dim=-1)
            
            retr_scores = (expert_scores * score_weights).sum(dim=-1)
            batch_retr_scores.append(retr_scores)
        return batch_retr_scores
  
    def recompute_fusion_score(self, fusion_states, batch_data):
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
            passage_scores = item_scores.sum(dim=[0,2])
            batch_passage_scores.append(passage_scores)
                 
        return batch_passage_scores, input_states 
         
    def get_fabric_input_states(self, opt_info, batch_data):
        input_states = opt_info['input_states']
        passage_num_lst = [len(a['passages']) for a in batch_data] 
        offset = 0
        batch_states = []
        for passage_num in passage_num_lst:
            pos = offset + passage_num
            item_input_states = input_states[offset:pos]
            batch_states.append(item_input_states)
            offset = pos
        
        return batch_states
    
    def compute_score_weights(self, fabric_states, fusion_states):
        input_states = torch.cat([fabric_states, fusion_states], dim=-1)
        scores = self.expert_fnt(input_states) 
        weights = F.softmax(scores, dim=-1)
        return weights 
