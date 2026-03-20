# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm, trange

# import a previous version of the HuggingFace Transformers package
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

IDENTITIES = {0: "Nam'er", 1: "Se'er", 2: "Shap'er", 3: "Hear'er"}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

import time
import logging
import torch
import torch.profiler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import AdamW, WarmupLinearSchedule

# Assuming logger is initialized at the top of your script
logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer):
    """ Train the model using Manual All-Reduce sync + Steady-State Monitoring """

    args.train_batch_size = args.per_device_train_batch_size
    
    # Ensure all nodes have the exact same number of steps
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=args.world_size, 
        rank=args.local_rank, 
        shuffle=True,
        drop_last=True    
    )

    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size
    )

    my_identity = IDENTITIES[args.local_rank]

    # Calculate optimization steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("[%s] ***** Running training *****", my_identity)
    
    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    
    # --- STATISTICS TRACKING ---
    loss_history = []
    time_history = [] 

    # --- PROFILER SETUP ---
    # wait=2 to match our timing trick (skipping first iteration in trace too)
    prof_schedule = torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1)
    
    def trace_handler(p):
        output = f"trace_allreduce_rank_{args.local_rank}.json"
        p.export_chrome_trace(output)
        logger.info(f"Rank {args.local_rank} profiling trace saved to {output}")

    set_seed(args)
    
    # Start the Profiler Context
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=False # Disabling stack for stability on CPU clusters
    ) as prof:

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        
        for epoch in train_iterator:
            # --- [TIMING TRICK]: INITIALIZE EPOCH TRACKERS ---
            epoch_start_time_after_warmup = None
            timed_steps_count = 0

            if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels':         batch[3]}
                
                outputs = model(**inputs)
                loss = outputs[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward pass
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                tr_loss += loss.item()

                # Optimization and All-Reduce step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    
                    # --- MANUAL ALL-REDUCE SYNC ---
                    local_grad = get_flat_grads(model)
                    torch.distributed.all_reduce(local_grad, op=torch.distributed.ReduceOp.SUM)
                    local_grad /= args.world_size
                    set_flat_grads(model, local_grad)

                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # --- [TIMING TRICK]: START CLOCK AFTER STEP 0 IS FULLY DONE ---
                    if step == 0:
                        # Step 0 included the first flattening, all_reduce, and optimizer call.
                        # We start the clock NOW to measure pure steady-state speed.
                        epoch_start_time_after_warmup = time.time()
                    elif epoch_start_time_after_warmup is not None:
                        # Increment count for iterations following iteration 0
                        timed_steps_count += 1

                    # Record synchronized loss
                    loss_history.append(loss.item())

                # Step the profiler
                prof.step()

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            # --- [TIMING TRICK]: CALCULATE STEADY-STATE AVERAGE ---
            if epoch_start_time_after_warmup is not None and timed_steps_count > 0:
                # Time from end of step 0 to the end of the epoch iterator
                steady_state_duration = time.time() - epoch_start_time_after_warmup
                # Average duration of the N-1 steps
                avg_time_per_it = steady_state_duration / timed_steps_count
                
                time_history.append((epoch, avg_time_per_it, steady_state_duration))
                
                logger.info("Rank %d | Epoch %d | Steady-state Avg: %.4f s | Timed Steps: %d", 
                            args.local_rank, epoch, avg_time_per_it, timed_steps_count)

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

    # --- FINAL: SAVE CSV DATA ---
    with open(f"loss_curve_allreduce_rank_{args.local_rank}.csv", "w") as f:
        f.write("step,loss\n")
        for i, l in enumerate(loss_history):
            f.write(f"{i},{l}\n")

    with open(f"timing_stats_allreduce_rank_{args.local_rank}.csv", "w") as f:
        f.write("epoch,avg_time_per_it,total_steady_state_time\n")
        for e, avg, total in time_history:
            f.write(f"{e},{avg:.6f},{total:.6f}\n")
            
    logger.info(f"Rank {args.local_rank} successfully saved all CSV statistics.")

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    my_identity = IDENTITIES[args.local_rank]
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        if args.local_rank == 0:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        
        for key in sorted(result.keys()):
            print(f"\nEval Result: \n{key}={str(result[key])}")

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def get_flat_params(model):
    # Get all parameters that actually require gradients
    params = [p.data for p in model.parameters()]
    # Flatten and concatenate into one giant 1D vector
    return torch.cat([p.view(-1) for p in params])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        # Slice the flat vector and copy the data back into the parameter
        p.data.copy_(flat_params[offset : offset + numel].view_as(p.data))
        offset += numel

def get_flat_grads(model):
    # 1. Collect only the gradients that exist (not all params have grads!)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1))
    
    # 2. Stitch them into one giant 1D tensor (approx 110M floats for BERT)
    return torch.cat(grads)

def set_flat_grads(model, flat_grads):
    offset = 0
    for p in model.parameters():
        if p.grad is not None:
            numel = p.grad.numel()
            # 3. Slice the giant vector and copy it back into the specific layer's grad buffer
            p.grad.data.copy_(flat_grads[offset : offset + numel].view_as(p.grad))
            offset += numel

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_device_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank. If single-node training, local_rank defaults to -1.")
    
    #Distributed Training Args
    parser.add_argument("--master_ip", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="23456")
    parser.add_argument("--world_size", type=int, default=4)

    args = parser.parse_args()

    print("")
    my_identity = IDENTITIES[args.local_rank]
    os.environ['MASTER_ADDR'] = args.master_ip
    os.environ['MASTER_PORT'] = args.master_port

    logger.info("***** Distributed Training Configuration *****")
    logger.info(f"  Process Rank:    {args.local_rank}")
    logger.info(f"  World Size:      {args.world_size}")
    logger.info(f"  Master IP:       {args.master_ip}")
    logger.info(f"  Master Port:     {args.master_port}")
    print(f"--rank = {args.local_rank}")

    logger.warning(
        "Process rank: %s, distributed: %s",
        args.local_rank,
        bool(args.world_size > 1)
    )

    print(f"RANK {args.local_rank} CURRENT DIRECTORY: {os.getcwd()}", flush=True)
    print(f"RANK {args.local_rank} LOG FILE PATH: {os.path.abspath(f'node_{args.local_rank}.log')}", flush=True)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # set up (distributed) training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, args.device, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 1. INITIALIZE FIRST (Essential)
    dist.init_process_group(
        backend="gloo",          
        rank=args.local_rank,          
        world_size=args.world_size 
    )

    # 2. COORDINATED LOADING
    # If not rank 0, wait for rank 0 to finish downloading/loading
    if args.local_rank != 0:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, 
        num_labels=num_labels, 
        finetuning_task=args.task_name
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, 
        do_lower_case=args.do_lower_case
    )

    # Load the model weights
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    print(f"Rank {args.local_rank}: loaded model")

    # Now Rank 0 hits the barrier to release the other ranks
    if args.local_rank == 0:
        torch.distributed.barrier()

    # 3. MANUAL SYNC (The "Master Truth" Broadcast)
    # Ensure the buffer is on the correct device (args.device)
    total_params = sum(p.numel() for p in model.parameters())

    if args.local_rank == 0:
        flat_weights = get_flat_params(model).to(args.device)
    else:
        flat_weights = torch.zeros(total_params).to(args.device)

    # Sync everyone to Rank 0's weights
    dist.broadcast(flat_weights, src=0)

    # Unpack
    set_flat_params(model, flat_weights)

    if args.local_rank != 0:
        logger.info("%s: Loaded Init Weights", my_identity)
    else:
        logger.info("%s: Sent Init Weights", my_identity)

    print(f"Rank {args.local_rank}: Fellowship synchronized. All models are bit-for-bit identical.")

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    evaluate(args, model, tokenizer, prefix="")

if __name__ == "__main__":
    main()
