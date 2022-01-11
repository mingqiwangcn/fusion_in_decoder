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
from ensemble_retr_model import EnsembleRetrModel 
from ensemble_retr_loss import EnsembleRetrLoss
from fabric_qa.answer_ranker.train import get_batch_data as get_fabric_examples
from fabric_qa.answer_ranker.train import clear_features
from fabric_qa.answer_ranker.train import get_model_answers, read_qas, save_model, get_open_model
import logging
from fabric_qa import utils as fabric_utils
import torch.optim as optim
import time

Num_Answers = 1
Train_Cache_Dir = '/home/cc/data/qa_zre_data/data/open_qa/train/data_5_percent/train_features'
Eval_Cache_Dir = '/home/cc/data/qa_zre_data/data/open_qa/dev/dev_features'
checkpoint_steps = 5000

def get_device(cuda):
    device = torch.device(("cuda:%d" % cuda) if torch.cuda.is_available() and cuda >=0 else "cpu")
    return device

def get_loss_fn(opt):
    loss_fn = EnsembleRetrLoss(opt.device)
    return loss_fn

def get_retr_model(opt):
    opt.num_best = 1
    reader_file = '/home/cc/code/fabric_qa/model/reader/forward_reader/model'
    open_model = get_open_model(opt, reader_file, model_id='model_1') 
    
    open_model_file = '/home/cc/code/fabric_qa/model/answer_ranker/qa_zre/epoc_8_model.pt'
    state_dict = torch.load(open_model_file, map_location=opt.device)
    open_model.load_state_dict(state_dict)
    
    retr_model = EnsembleRetrModel(open_model)
    retr_model = retr_model.to(opt.device) 
    return retr_model

def log_metrics(epoc, metric_rec,
                batch_data, batch_score, batch_answers, 
                time_span, total_time,
                itr, num_batch, 
                loss=None, model_tag=None):
    batch_size = len(batch_score)
    batch_sorted_idxes = []
    answer_num_lst = []
    for b_idx in range(batch_size):
        item_scores = batch_score[b_idx]
        answer_scores = item_scores 

        scores = answer_scores.data.cpu().numpy()
        sorted_idxes = np.argsort(-scores)

        answer_lst = []
        item_answer_lst = batch_answers[b_idx]
        for answer_info in item_answer_lst:
            answer_lst.extend(answer_info)

        ems = [answer_lst[idx]['em'] for idx in sorted_idxes]
        f1s = [answer_lst[idx]['f1'] for idx in sorted_idxes]

        metric_rec.update(ems, f1s)
        batch_sorted_idxes.append(sorted_idxes)
        answer_num_lst.append(len(sorted_idxes))

    max_answer_num = max(answer_num_lst)
    metric_mean = metric_rec.get_mean()
    str_info = ('epoc=%d ' % epoc) if epoc is not None else ''
    if model_tag is not None:
        str_info += '%s ' % model_tag
    if not loss is None:
        str_info += 'loss=%.6f ' % loss

    str_info += '(em, f1)=(%.2f, %.2f) max-top-%d(em, f1)=(%.2f, %.2f) time=%.2f total=%.2f %d/%d'\
             % (metric_mean[0], metric_mean[1], max_answer_num, metric_mean[2], metric_mean[3],
                time_span, total_time, itr, num_batch)
    logger.info(str_info)

    return batch_sorted_idxes


def evaluate(epoc, model, retr_model, dataset, dataloader, eval_qas, tokenizer, opt, model_tag=None):
    model.eval()
    retr_model.eval()
   
    metric_rec = fabric_utils.MetricRecorder() 
    total_time = .0 
    model.overwrite_forward_crossattention()
    model.reset_score_storage()
    with torch.no_grad():
        num_batch = len(dataloader)
        for itr, fusion_batch in tqdm(enumerate(dataloader), total=num_batch):
            t1 = time.time()

            scores, score_states, examples = get_score_info(model, fusion_batch, dataset)
            batch_fabric_data, batch_fabric_examples = get_fabric_batch_data(examples, Eval_Cache_Dir)
            retr_scores = retr_model(batch_fabric_data, batch_fabric_examples, scores, score_states)    
            clear_features(batch_fabric_data)
            batch_answers = get_batch_answers(batch_fabric_data, eval_qas) 
            
            t2 = time.time()
            time_span = t2 - t1
            total_time += time_span
           
            if ((itr + 1) % 10 == 0) or (itr + 1 == num_batch):  
                log_metrics(epoc, metric_rec, batch_fabric_data, retr_scores, batch_answers, time_span, total_time, 
                            (itr + 1), num_batch, loss=None, model_tag=model_tag) 
            
def train(model, retr_model, 
          train_dataset, train_qas, collator,
          eval_dataset, eval_dataloader, eval_qas_data, 
          tokenizer, 
          opt):

    metric_rec = fabric_utils.MetricRecorder()
    loss_fn = get_loss_fn(opt)
    learing_rate = 1e-3
    optimizer = optim.Adam(retr_model.parameters(), lr=learing_rate)

    global_steps = 0
    max_epoc = 10
    total_time = .0
    for epoc in range(max_epoc):
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

            scores, score_states, examples = get_score_info(model, fusion_batch, train_dataset)
            batch_fabric_data, batch_fabric_examples = get_fabric_batch_data(examples, Train_Cache_Dir)
            retr_scores = retr_model(batch_fabric_data, batch_fabric_examples, scores, score_states)    
            clear_features(batch_fabric_data)
            batch_answers = get_batch_answers(batch_fabric_data, train_qas) 
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
            
            if ((itr + 1) % 10 == 0) or (itr + 1 == num_batch):  
                log_metrics(epoc, metric_rec, batch_fabric_data, retr_scores, batch_answers, time_span, total_time, 
                            (itr + 1), num_batch, loss.item()) 
            
            if global_steps % checkpoint_steps == 0:
                model_tag = 'step_%d' % global_steps
                out_dir = os.path.join(opt.checkpoint_dir, opt.name)
                save_model(out_dir, retr_model, epoc, tag=model_tag)
                evaluate(epoc, model, retr_model,
                         eval_dataset, eval_dataloader, eval_qas_data,
                         tokenizer, opt, model_tag=model_tag)
                retr_model.train() 

def get_fabric_batch_data(fusion_examples, cache_dir):
    batch_fabric_data = []
    for f_example in fusion_examples:
        ctx_data = f_example['ctxs']
        item_data = {
            'qid':f_example['id'],
            'question':f_example['question'],
            'passages':[a['text'] for a in ctx_data],
            'p_id_lst':[a['p_id'] for a in ctx_data]
        }
        batch_fabric_data.append(item_data)
    
    _, batch_fabric_examples = get_fabric_examples(batch_fabric_data, cache_dir)
    return batch_fabric_data, batch_fabric_examples 
    
def get_score_info(model, batch_data, dataset):
    with torch.no_grad():
        (idx, _, _, context_ids, context_mask) = batch_data
        model.reset_score_storage()
        outputs = model.generate(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            max_length=50,
            num_beams=Num_Answers,
            num_return_sequences=Num_Answers
        )
        crossattention_scores, score_states = model.get_crossattention_scores(context_mask.cuda())
        batch_examples = [] 
        for k, _ in enumerate(outputs):
            example = dataset.data[idx[k]]
            batch_examples.append(example)
        
    return crossattention_scores, score_states, batch_examples
    
def write_preds(qid, top_passage_idx, top_passage):
    out_item = {
        'qid':qid,
        'top_passage_id':int(top_passage_idx),
        'top_passage':top_passage
    }
    f_o_preds.write(json.dumps(out_item) + '\n')

def show_precision(count, table_pred_results):
    str_info = 'count = %d' % count
    for threshold in table_pred_results:
        precision = np.mean(table_pred_results[threshold]) * 100
        str_info += 'p@%d = %.2f ' % (threshold, precision)
    logger.info(str_info)

def get_batch_answers(batch_data, qas_data):
    batch_answers = get_model_answers(batch_data, qas_data, 'model_1')
    return batch_answers

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
   
    global f_o_preds 
    out_preds_file = os.path.join(opt.checkpoint_dir, opt.name, 'preds.jsonl')
    f_o_preds = open(out_preds_file, 'w')

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
    eval_qas_data = read_qas(opt.eval_qas_file)
    
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
        train_qas_data = read_qas(opt.train_qas_file)
        logger.info("Start train")
        train(model, retr_model, 
              train_dataset, train_qas_data, collator_function,
              eval_dataset, eval_dataloader, eval_qas_data, 
              tokenizer, opt)
    else:
        logger.info("Start eval")
        evaluate(model, retr_model,
                eval_dataset, eval_dataloader, eval_qas_data,
                tokenizer, opt)
     
    f_o_preds.close()

    #logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

if __name__ == "__main__":
    main()
