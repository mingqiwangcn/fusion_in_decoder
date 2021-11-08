# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import src.coherence_model as coherence
from tqdm import tqdm

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        #num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    num_batch = len(train_dataloader)
    while step < opt.total_steps:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader), total=num_batch):
            (idx, labels, _, context_ids, context_mask, batch_examples) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                batch_examples=batch_examples
            )

            if train_loss is None:
                model.zero_grad()
                continue
                
            step += 1
            
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    curr_loss = 0
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    with torch.no_grad():
        num_batch = len(dataloader)
        for i, batch in tqdm(enumerate(dataloader), total=num_batch):
            (idx, _, _, context_ids, context_mask, batch_examples) = batch

            model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=None,
                batch_examples=batch_examples
            )
            
            for b_idx, example in enumerate(batch_examples):
                top_preds = example['src_data_item']['top_preds']
                coherence_score_lst = [a['coherence_score'] for a in top_preds]
                best_idx = np.argmax(coherence_score_lst)
                em_score = top_preds[best_idx]['em']
                total += 1
                exactmatch.append(em_score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

def create_model(options, tokenizer, collator):
    f_reader_model = src.model.FiDT5.from_pretrained(opt.f_reader_model_path)
    f_reader_model = f_reader_model.to(opt.device)
    f_reader = coherence.ForwardReader(tokenizer, f_reader_model)

    b_reader_model = src.model.FiDT5.from_pretrained(opt.b_reader_model_path)
    b_reader_model = b_reader_model.to(opt.device)
    b_reader = coherence.BackwardReader(tokenizer, f_reader_model)
   
    model = coherence.CoherenceModel(f_reader, b_reader, collator)
    model = model.to(opt.device)
    return model

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()

    options.parser.add_argument('--f_reader_model_path', type=str)
    options.parser.add_argument('--b_reader_model_path', type=str)

    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    #model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = coherence.CoherenceCollator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    model = create_model(options, tokenizer, collator)
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0 
    '''
    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)
    '''

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
