# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import os
import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data

from src.ondisk_index import OndiskIndexer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def retrieve_data(opt, index, data, model, tokenizer, f_o):
    batch_size = 1
    dataset = src.data.Dataset(data, n_context=1)
    collator = src.data.Collator(opt.question_maxlength, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            (idxes, _, _, question_ids, question_mask, _) = batch
            out_emb = model.embed_text(
                text_ids=question_ids.to(opt.device).view(-1, question_ids.size(-1)), 
                text_mask=question_mask.to(opt.device).view(-1, question_ids.size(-1)), 
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            query_emb = out_emb.cpu().numpy()
            result_lst = index.search(query_emb, top_n=opt.n_docs, n_probe=128, batch_size=1)
            assert(1 == len(result_lst))
            item_result = result_lst[0]
            data_item = data[idxes[0]]
            ctxs_num = len(item_result)
            data_item['ctxs'] =[
                {
                    'id': int(item_result[c]['p_id']),
                    'title': '',
                    'text': item_result[c]['passage'],
                    'score': float(item_result[c]['score']),
                    'tag':item_result[c]['tag']
                } for c in range(ctxs_num)
            ]
            f_o.write(json.dumps(data_item) + '\n') 

    #logger.info(f'Questions embeddings shape: {embedding.size()}')

def main(opt):
    if os.path.exists(args.output_path):
        print('[%s] already exists' % args.output_path)
        return
    
    src.util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = src.data.load_data(opt.data)
    model_class = src.model.Retriever
    model = model_class.from_pretrained(opt.model_path)

    model.cuda()
    model.eval()
    #if not opt.no_fp16:
        #model = model.half()

    index = OndiskIndexer(args.index_file, args.passage_file)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
     
    with open(args.output_path, 'w') as f_o:
        retrieve_data(opt, index, data, model, tokenizer, f_o)

    logger.info(f'Saved results to {args.output_path}')

def read_passages(data_file):
    passages = []
    passage_file_lst = glob.glob(data_file)
    for passage_file in passage_file_lst:
        part_passages = src.util.load_passages(passage_file)
        passages.extend(part_passages)
    return passages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--index_file', type=str)
    parser.add_argument('--passage_file', type=str)

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help=".json file containing question and answers, similar format to reader data")
    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.tsv file)')
    parser.add_argument('--passages_embeddings', type=str, default=None, help='Glob path to encoded passages')
    parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    parser.add_argument('--n-docs', type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument('--validation_workers', type=int, default=32,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--per_gpu_batch_size', type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action='store_true', 
                        help='If enabled, save index and load index if it exists')
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--question_maxlength', type=int, default=40, help="Maximum number of tokens in a question")
    parser.add_argument('--indexing_batch_size', type=int, default=50000, help="Batch size of the number of passages indexed")
    parser.add_argument("--n-subquantizers", type=int, default=0, 
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n-bits", type=int, default=8, 
                        help='Number of bits per subquantizer')


    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
