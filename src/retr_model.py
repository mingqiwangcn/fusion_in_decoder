import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionRetrModel(nn.Module):
    def __init__(self):
        super(FusionRetrModel, self).__init__()
        D = 768
        
        self.passage_fnt = nn.Linear(D * 3, D)

        self.table_fnt = nn.Linear(D * 3, D)

        self.feature_fnt = nn.Sequential(
                        nn.Linear(D * 2, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
        )
         
    def forward(self, batch_data, fusion_scores, fusion_states, passage_masks, opts=None):
        fusion_scores_redo = self.recompute_fusion_score(batch_data, fusion_scores, fusion_states, passage_masks,
                             opts=opts)
        return fusion_scores_redo
 
    def std_norm(self, scores):
        mean_score = scores.mean()
        std_score = scores.std()
        ret_scores = (scores - mean_score) / (std_score + 1e-5)
        return ret_scores

    def get_table_aggr_states(self, batch_data, batch_input_states, opts=None):
        batch_passage_states = self.passage_fnt(batch_input_states)
        bsz, num_layers, _, num_feature_1 = batch_passage_states.shape
        batch_p_table_states = self.table_fnt(batch_input_states) 
        _, _, _, num_feature_2 = batch_p_table_states.shape
         
        item_passage_feature_lst = []
        for idx, item in enumerate(batch_data):
            num_passages = len(item['passages'])
            passage_states = batch_passage_states[idx].view(num_layers, num_passages, -1, num_feature_1)
            p_table_states = batch_p_table_states[idx].view(num_layers, num_passages, -1, num_feature_2)
            tag_lst = item['tags']
            table_feature_dict = {}
            aggr_feature_dict = {}
            p_table_lst = []
            for passage_idx, tag in enumerate(tag_lst):
                table_id = tag['table_id']
                p_table_lst.append(table_id)
                if table_id not in table_feature_dict:
                    table_feature_dict[table_id] = []
                
                p_table_feature = p_table_states[:, passage_idx:(passage_idx+1), :, :]
                table_feature_lst = table_feature_dict[table_id]
                table_feature_lst.append(p_table_feature)
            
            for p_table_id in table_feature_dict:
                p_table_feature_lst = table_feature_dict[p_table_id]
                table_features = torch.cat(p_table_feature_lst, dim=1)
                table_aggr_feature = table_features.max(dim=1, keepdim=True)[0]
                aggr_feature_dict[p_table_id] = table_aggr_feature

                table_reg_score = self.compute_reg_score(table_features)
                reg_score_lst = opts.get('reg_score', None)
                if reg_score_lst is None:
                    opts['reg_score'] = []
                reg_score_lst = opts['reg_score']
                reg_score_lst.append(table_reg_score)

            p_aggr_feature_lst = []
            for table_id in p_table_lst:
                aggr_feature = aggr_feature_dict[table_id]
                p_aggr_feature_lst.append(aggr_feature)
            p_aggr_features = torch.cat(p_aggr_feature_lst, dim=1) 

            item_passage_feature = torch.cat([passage_states, p_aggr_features], dim=-1)
            _, _, _, num_aggr_features = item_passage_feature.shape
            item_passage_feature = item_passage_feature.view(num_layers, -1, num_aggr_features)
            item_passage_feature_lst.append(item_passage_feature.unsqueeze(0))
       
        batch_passage_features = torch.cat(item_passage_feature_lst, dim=0)
        return batch_passage_features 

    def compute_reg_score(self, table_features):
        n_layer, n_passages, n_tokens, n_feature = table_features.shape
        states = table_features.mean(dim=(0,2))
        scores = torch.mm(states, states.t())
        reg_score = torch.triu(scores, diagonal=1).mean() 
        return reg_score

    def recompute_fusion_score(self, batch_data, fusion_scores, fusion_states, passage_masks, opts=None):
        answer_states = fusion_states['answer_states']
        answer_states = answer_states[:, -1:, :, :]
        bsz, n_layers, _, emb_size = answer_states.size()
        query_passage_states = fusion_states['query_passage_states'] 
        query_passage_states = query_passage_states[:, -1:, :, :]
        _, _, n_tokens, _ = query_passage_states.size()
        
        answer_states = answer_states.expand(bsz, n_layers, n_tokens, emb_size)
        input_states = [answer_states, query_passage_states, answer_states * query_passage_states]
        input_states = torch.cat(input_states, dim=-1)
       
        p_aggr_features = self.get_table_aggr_states(batch_data, input_states, opts=opts)
        batch_scores = self.feature_fnt(p_aggr_features).squeeze(-1)
       
        batch_passage_scores = []
        for idx in range(len(batch_data)):
            n_passages = len(batch_data[idx]['passages'])
            item_masks = passage_masks[idx].expand(n_layers, -1, -1)
            item_scores = batch_scores[idx].view(n_layers, n_passages, -1)
            item_masked_scores = item_scores * item_masks
            item_adapt_scores = item_masked_scores.sum(dim=[0,2])
            item_fusion_scores = fusion_scores[idx]
            passage_scores = item_adapt_scores # + item_fusion_scores
            batch_passage_scores.append(passage_scores)
                 
        return batch_passage_scores 
         
