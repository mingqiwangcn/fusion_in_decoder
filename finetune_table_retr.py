# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from tqdm import tqdm
import os
import json
from src.retr_model import FusionRetrModel 
from src.retr_loss import FusionRetrLoss

from src.general_retr_model import FusionGeneralRetrModel
from src.general_retr_loss import FusionGeneralRetrLoss

import logging
import torch.optim as optim
import time
from fabric_qa import utils as fabric_utils

Num_Answers = 1
global_steps = 0

def get_device(cuda):
    device = torch.device(("cuda:%d" % cuda) if torch.cuda.is_available() and cuda >=0 else "cpu")
    return device

def get_loss_fn(opt):
    if (opt.retr_model_type is None) or (opt.retr_model_type == ''):
        loss_fn = FusionRetrLoss()
        logger.info('loss function, FusionRetrLoss')
    else: 
        loss_fn = FusionGeneralRetrLoss()
        logger.info('loss function, FusionGeneralRetrLoss')

    return loss_fn

def get_retr_model(opt):
    if (opt.retr_model_type is None) or (opt.retr_model_type == ''):
        retr_model = FusionRetrModel()
        logger.info('retr_model, FusionRetrModel')
    else:
        retr_model = FusionGeneralRetrModel()
        logger.info('retr_model, FusionGeneralRetrModel')
    if opt.fusion_retr_model is not None:
        state_dict = torch.load(opt.fusion_retr_model, map_location=opt.device)
        retr_model.load_state_dict(state_dict)

    retr_model = retr_model.to(opt.device) 
    return retr_model

def write_predictions(f_o_pred, item, top_idxes):
    qid = item['qid']
    table_id_lst = item['table_id_lst']
    question = item['question']
    passages = item['passages']
    tags = item['tags']
    out_item = {
        'qid':qid,
        'table_id_lst':table_id_lst,
        'question':question,
        'passages':[passages[a] for a in top_idxes],
        'tags':[tags[a] for a in top_idxes]
    }
    f_o_pred.write(json.dumps(out_item) + '\n')

def log_metrics(epoc, metric_rec,
                batch_data, batch_score, batch_answers, 
                time_span, total_time,
                itr, num_batch, 
                loss=None, model_tag=None, f_o_pred=None):
    batch_size = len(batch_score)
    batch_sorted_idxes = []
    answer_num_lst = []
    for b_idx in range(batch_size):
        item_scores = batch_score[b_idx]
        answer_scores = item_scores 
        scores = answer_scores.data.cpu().numpy()
        top_m = min(len(scores), 30)
        sorted_idxes = np.argpartition(-scores, range(top_m))[:top_m]
        item_answer_lst = batch_answers[b_idx]
        #assert(len(sorted_idxes) == len(item_answer_lst))
        ems = [item_answer_lst[idx]['em'] for idx in sorted_idxes]
        f1s = ems
        metric_rec.update(ems, f1s)
        batch_sorted_idxes.append(sorted_idxes)
        answer_num_lst.append(len(sorted_idxes))
        
        if f_o_pred is not None:
            write_predictions(f_o_pred, batch_data[b_idx], sorted_idxes)

    max_answer_num = max(answer_num_lst)
    metric_mean = metric_rec.get_mean()
    str_info = ('epoc=%d ' % epoc) if epoc is not None else ''
    if loss is None:
        str_info = ' ' + str_info
    if model_tag is not None:
        str_info += '%s ' % model_tag
    if not loss is None:
        str_info += 'loss=%.6f ' % loss

    str_info += 'p@1=%.2f p@%d=%.2f time=%.2f total=%.2f %d/%d'\
             % (metric_mean[0], max_answer_num, metric_mean[2], time_span, total_time, itr, num_batch)
    
    if loss is not None:
        str_info += ' global_steps=%d' % global_steps
    logger.info(str_info)

    return batch_sorted_idxes


def evaluate(epoc, model, retr_model, dataset, dataloader, tokenizer, opt, model_tag=None, out_dir=None):
    logger.info('Start evaluation')
    model.eval()
    retr_model.eval()
  
    f_o_pred = None
    if out_dir is not None:
        out_pred_file = os.path.join(out_dir, 'pred_%s.jsonl' % model_tag)
        f_o_pred = open(out_pred_file, 'w')
   
    metric_rec = fabric_utils.MetricRecorder() 
    total_time = .0 
    model.overwrite_forward_crossattention()
    model.reset_score_storage()
    with torch.no_grad():
        num_batch = len(dataloader)
        for itr, fusion_batch in tqdm(enumerate(dataloader), total=num_batch):
            t1 = time.time()

            scores, score_states, examples, context_mask = get_score_info(model, fusion_batch, dataset)
            batch_data = get_batch_data(examples)
            retr_scores = retr_model(batch_data, scores, score_states, context_mask)    
            batch_answers = get_batch_answers(batch_data)
             
            t2 = time.time()
            time_span = t2 - t1
            total_time += time_span
           
            log_metrics(epoc, metric_rec, batch_data, retr_scores, batch_answers, time_span, total_time, 
                        (itr + 1), num_batch, loss=None, model_tag=model_tag, f_o_pred=f_o_pred) 
    
    if f_o_pred is not None:
        f_o_pred.close() 
            
def train(model, retr_model, 
          train_dataset, collator,
          eval_dataset, eval_dataloader, 
          tokenizer, 
          opt):

    loss_fn = get_loss_fn(opt)
    learing_rate = 1e-3
    optimizer = optim.Adam(retr_model.parameters(), lr=learing_rate)
    
    global global_steps
    global_steps = 0
    max_epoc = 10
    total_time = .0
    for epoc in range(max_epoc):
        metric_rec = fabric_utils.MetricRecorder()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=opt.per_gpu_batch_size,
            drop_last=True,
            #num_workers=10,
            collate_fn=collator
        )

        model.eval()
        if hasattr(model, "module"):
            model = model.module

        model.overwrite_forward_crossattention()
        model.reset_score_storage() 

        assert(Num_Answers == 1) 

        table_pred_results = {1:[], 5:[]}
        count = 0
        num_batch = len(train_dataloader)
        for itr, fusion_batch in tqdm(enumerate(train_dataloader), total=num_batch):
            t1 = time.time()

            scores, score_states, examples, context_mask = get_score_info(model, fusion_batch, train_dataset)
            batch_data = get_batch_data(examples)
            retr_scores = retr_model(batch_data, scores, score_states, context_mask) 
            batch_answers = get_batch_answers(batch_data) 
            loss = loss_fn(retr_scores, batch_answers)
            optimizer.zero_grad()
            if loss is None:
                continue
            loss.backward()
            optimizer.step()

            global_steps += 1
            
            t2 = time.time()
            time_span = t2 - t1
            total_time += time_span
           
            
            log_metrics(epoc, metric_rec, batch_data, retr_scores, batch_answers, time_span, total_time, 
                        (itr + 1), num_batch, loss.item()) 
            
            if global_steps % opt.checkpoint_steps == 0:
                model_tag = 'step_%d' % global_steps
                out_dir = os.path.join(opt.checkpoint_dir, opt.name)
                save_model(out_dir, retr_model, epoc, tag=model_tag) 
                show_tag = 'step=%d' % global_steps
                evaluate(epoc, model, retr_model,
                         eval_dataset, eval_dataloader,
                         tokenizer, opt, model_tag=show_tag, out_dir=out_dir)
                retr_model.train() 

def get_batch_answers(batch_data):
    batch_answers = []
    for item in batch_data:
        answers = item['answers']
        batch_answers.append(answers)
    return batch_answers

def save_model(output_dir, model, epoc, tag='step'):
    file_name = 'epoc_%d_%s_model.pt' % (epoc, tag)
    out_path = os.path.join(output_dir, file_name)
    torch.save(model.state_dict(), out_path) 

def get_batch_data(fusion_examples):
    batch_data = []
    for f_example in fusion_examples:
        gold_table_lst = f_example['table_id_lst']
        ctx_data = f_example['ctxs']
        
        item_answers = []
        for passage_info in ctx_data:
            answer_info = {
                'em':int(passage_info['tag']['table_id'] in gold_table_lst)
            }
            item_answers.append(answer_info)
        
        item_data = {
            'qid':f_example['id'],
            'table_id_lst':gold_table_lst,
            'question':f_example['question'],
            'passages':[a['text'] for a in ctx_data],
            'p_id_lst':[a['id'] for a in ctx_data],
            'answers':item_answers,
            'tags':[a['tag'] for a in ctx_data]
        }
        batch_data.append(item_data)
    
    return batch_data 
    
def get_score_info(model, batch_data, dataset):
    # context_ids, (bsz, num_passages, num_tokens)
    (batch_index_data, _, _, batch_passage_id_data, batch_passage_mask_data, batch_passage_tag_data) = batch_data
  
    batch_attention_scores = []
    batch_answer_states = []
    batch_passage_states = []
    batch_examples = []
     
    for item_pos, item_index in enumerate(batch_index_data):
        table_data_dict = {}
        passage_id_data = batch_passage_id_data[item_pos]
        passage_mask_data = batch_passage_mask_data[item_pos]
        passage_tag_data = batch_passage_tag_data[item_pos]

        for pos, tag in enumerate(passage_tag_data):
            table_id = tag['table_id']
            if table_id not in table_data_dict:
                table_data_dict[table_id] = {'id':[], 'mask':[], 'pos':[]}
            
            id_data_lst = table_data_dict[table_id]['id']
            id_data_lst.append(passage_id_data[pos])
            masK_data_lst = table_data_dict[table_id]['mask']
            masK_data_lst.append(passage_mask_data[pos])
            pos_lst = table_data_dict[table_id]['pos']
            pos_lst.append(pos) 
       
        num_passages = len(passage_tag_data)
        item_attention_scores = [None for a in range(num_passages)]
        item_answer_states = [None for a in range(num_passages)]
        item_passage_states = [None for a in range(num_passages)]

        for table_id in table_data_dict:
            id_lst = table_data_dict[table_id]['id']
            group_id_data = torch.stack(id_lst) 
            mask_lst = table_data_dict[table_id]['mask']
            group_mask_data = torch.stack(mask_lst)
            
            group_idxes = item_index.unsqueeze(0)
            group_id_data = group_id_data.unsqueeze(0)
            group_mask_data = group_mask_data.unsqueeze(0)
    
            group_data = (group_idxes, group_id_data, group_mask_data)
            attention_scores, score_states = get_table_score_info(model, group_data, dataset)
            attention_scores = attention_scores.squeeze(0)
            
            answer_states = score_states['answer_states'].squeeze(0)
            assert(answer_states.shape[1] == 1)
            answer_states = answer_states.unsqueeze(2)
            #answer_states  (n_layers, 1, 1, n_features)
            
            query_passage_states = score_states['query_passage_states'].squeeze(0)
            n_layers, _, n_features = query_passage_states.shape

            n_table_passages = group_mask_data.shape[1]
            n_max_tokens = group_mask_data.shape[2]
            passage_states = query_passage_states.view(n_layers, n_table_passages, -1, n_features)
             
            #passage_states (n_layers, n_passage, n_tokens, n_features)

            pos_lst = table_data_dict[table_id]['pos']
            for offset, pos in enumerate(pos_lst):
                item_attention_scores[pos] = attention_scores[offset].unsqueeze(0)
                item_answer_states[pos] = answer_states.expand(-1, -1, n_max_tokens, -1)
                item_passage_states[pos] = passage_states[:,offset:(offset+1),:,:] 
        
        item_attention_scores = torch.cat(item_attention_scores, dim=0).unsqueeze(0)
        batch_attention_scores.append(item_attention_scores)
        
        item_answer_states = torch.cat(item_answer_states, dim=1).view(n_layers, -1, n_features).unsqueeze(0)
        batch_answer_states.append(item_answer_states)
         
        item_passage_states = torch.cat(item_passage_states, dim=1).view(n_layers, -1, n_features).unsqueeze(0) 
        batch_passage_states.append(item_passage_states)
        
        item_example = dataset.data[item_index]
        item_example['ctxs'] = item_example['ctxs'][:num_passages]
        batch_examples.append(item_example) 
    
    batch_attention_scores = torch.cat(batch_attention_scores, dim=0)
    batch_answer_states = torch.cat(batch_answer_states, dim=0)
    batch_passage_states = torch.cat(batch_passage_states, dim=0)
    score_states = {'answer_states': batch_answer_states, 'query_passage_states': batch_passage_states}
    batch_masks = batch_passage_mask_data.to(batch_attention_scores.device) 
    return (batch_attention_scores, score_states, batch_examples, batch_masks)

def get_table_score_info(model, batch_data, dataset):
    with torch.no_grad():
        (idx, context_ids, context_mask) = batch_data
        model.reset_score_storage()
        outputs = model.generate(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            max_length=50,
            num_beams=Num_Answers,
            num_return_sequences=Num_Answers
        )
        crossattention_scores, score_states = model.get_crossattention_scores(context_mask.cuda())
    return crossattention_scores, score_states 
    

def show_precision(count, table_pred_results):
    str_info = 'count = %d' % count
    for threshold in table_pred_results:
        precision = np.mean(table_pred_results[threshold]) * 100
        str_info += 'p@%d = %.2f ' % (threshold, precision)
    logger.info(str_info)

def set_logger(opt):
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.propagate = False

    console = logging.StreamHandler()
    logger.addHandler(console)

    file_path = os.path.join(opt.checkpoint_dir, opt.name, 'log.txt')
    file_hander = logging.FileHandler(file_path, 'w')
    logger.addHandler(file_hander)

def main():
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if directory_exists:
        print('[%s] already exists.' % str(dir_path))
        return
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    global logger
    set_logger(opt)
    
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
   
    device = get_device(opt.cuda)
    opt.device = device
    
    retr_model = get_retr_model(opt)

    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size,
    )
    
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context,
        sort_by_score=False 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=0, 
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    if opt.do_train:
        train_examples = src.data.load_data(
            opt.train_data,
            global_rank=opt.global_rank,
            world_size=opt.world_size,
        )
        train_dataset = src.data.Dataset(train_examples, opt.n_context, sort_by_score=False)
        logger.info("Start train")
        train(model, retr_model, 
              train_dataset, collator_function,
              eval_dataset, eval_dataloader, 
              tokenizer, opt)
    else:
        logger.info("Start eval")
        out_dir = os.path.join(opt.checkpoint_dir, opt.name)
        evaluate(0, model, retr_model,
                eval_dataset, eval_dataloader,
                tokenizer, opt, out_dir=out_dir)

    #logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

if __name__ == "__main__":
    main()
