import torch
import torch.nn as nn
import torch.nn.functional as F
import src.data
from torch.utils.data import DataLoader, SequentialSampler

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
    
    def merge_passages(self, batch_data, batch_passage_scores):
        batch_merged_data = []
        for item_idx, item in enumerate(batch_data):
            passage_scores = batch_passage_scores[item_idx]
            passage_lst = item['passages']
            tag_lst = item['tags']
            table_passage_dict = {}
            for p_idx, passage in enumerate(passage_lst):
                tag_info = tag_lst[p_idx]
                table_id = tag_info['table_id']
                if table_id not in table_passage_dict:
                    table_passage_dict[table_id] = []
                
                table_passage_lst = table_passage_dict[table_id]
                passage_info = {'passage':passage, 'score':passage_scores[p_idx], 'tag':tag_info}
                table_passage_lst.append(passage_info) 
            
            item_merged_lst = self.merge_by_table(table_passage_dict)
            
            item_merged_data = {
                'qid':item['qid'],
                'question':item['question'],
                'table_id_lst':item['table_id_lst'],
                'answers':item['answers'],
                'passages':item_merged_lst
            }
            batch_merged_data.append(item_merged_data)
             
        return batch_merged_data  
    
    def merge_by_table(self, table_passage_dict):
        merged_lst = []
        for table_id in table_passage_dict:
            table_passage_lst = table_passage_dict[table_id]
            score_lst = [a['score'] for a in table_passage_lst]
            scores = torch.stack(score_lst)
            sorted_idxes = torch.argsort(-scores)
            top_idxes = sorted_idxes[:3] 
            top_passage_lst = [table_passage_lst[a] for a in top_idxes]
            #merge small passages by row and col
            
            merged_passage = ' '.join([a['passage'] for a in top_passage_lst])
            merged_tags = [a['tag'] for a in top_passage_lst]
            merged_info = {
                'passage':merged_passage,
                'tag':merged_tags,
                'table_id':table_id
            }
            merged_lst.append(merged_info) 
        return merged_lst 
     
    def forward(self, batch_data, fusion_scores, fusion_states, opt_info):
        fusion_scores_redo = self.recompute_fusion_score(batch_data, fusion_scores, fusion_states)
        batch_merged_data = self.merge_passages(batch_data, fusion_scores_redo)
        merged_scores = self.compute_merged_scores(batch_merged_data, opt_info)
        return merged_scores

    def compute_merged_scores(self, batch_merged_data, opt_info):
        reader_examples = []
        passage_num_lst = []
        for item in batch_merged_data:
            reader_item = {}
            reader_item['id'] = item['qid']
            reader_item['question'] = item['question']
            reader_item['table_id_lst'] = item['table_id_lst']
            reader_item['answers'] = item['answers']
            merged_passages = item['passages']
            reader_item['ctxs'] =[
                {
                    'id': merged_p_id,
                    'title': '',
                    'text': p_info['passage'],
                    'score':1,
                    'tag':p_info['tag']
                } for merged_p_id, p_info in enumerate(merged_passages)
            ]
            passage_num_lst.append(len(reader_item['ctxs']))
            reader_examples.append(reader_item) 
       
        N = max(passage_num_lst)
        reader_dataset = src.data.Dataset(reader_examples, N, sort_by_score=False) 
        sampler = SequentialSampler(reader_dataset)
        data_loader = DataLoader(reader_dataset, 
                                 sampler=sampler, 
                                 batch_size=len(batch_merged_data),
                                 num_workers=0,
                                 collate_fn=opt_info['collator'])
       
        get_score_info_func = opt_info['get_score_info']
        reader_model = opt_info['reader'] 
        assert(len(data_loader) == 1)
        fusion_data = None
        for fusion_batch in data_loader:
            fusion_data = fusion_batch
            break
        fusion_scores, fusion_states, _ = get_score_info_func(reader_model, fusion_data, reader_dataset)
        merged_scores = self.recompute_fusion_score(batch_merged_data, fusion_scores, fusion_states) 
        opt_info['merged_data'] = batch_merged_data 
        return merged_scores
     
    def std_norm(self, scores):
        mean_score = scores.mean()
        std_score = scores.std()
        ret_scores = (scores - mean_score) / (std_score + 1e-5)
        return ret_scores

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
            item_adapt_scores = item_scores.sum(dim=[0,2])
            item_fusion_scores = fusion_scores[idx]
            passage_scores = self.std_norm(item_adapt_scores) # + self.std_norm(item_fusion_scores) 
            batch_passage_scores.append(passage_scores)
                 
        return batch_passage_scores 
         
