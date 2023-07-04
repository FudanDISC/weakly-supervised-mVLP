# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
from __future__ import absolute_import, division, print_function
import argparse
from collections import defaultdict
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import base64
import os.path as op
import random, json
from ssl import _create_unverified_context
import numpy as np
import torch
import glob
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml
from albef import utils as albef_utils
import time, pickle

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed, weighted_sample
from albef.modeling.model_vqa import ALBEF_CLS, ALBEF_GEN, ALBEFforClassification
from transformers_past.pytorch_transformers import BertTokenizer, BertConfig 
from transformers_past.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers_past.pytorch_transformers.modeling_utils import WEIGHTS_NAME
from PIL import Image
from transformers import AutoTokenizer
from albef.modeling.vit import interpolate_pos_embed
from albef.modeling.transforms_utils import create_transform
from albef.optim import create_optimizer
from pathlib import Path

WEIGHTS_NAME = 'ckpt.pth'
VNLI_labelmap = {"contradiction": 0, "entailment": 1, "neutral": 2}

class xVNLIDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    # GQA dataset
    def __init__(self, caption_file, img_transform, tokenizer, args, is_train=True):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(xVNLIDataset, self).__init__()
        self.image_dir = args.image_dir
        self.img_transform = img_transform
        
        self.data = []
        logger.info('loading dataset from {} with images from {}'.format(caption_file, args.image_dir))
        for line in open(caption_file, 'r'):
            info = json.loads(line)
            if info['gold_label'] not in VNLI_labelmap:
                continue
            # assert info['gold_label'] in VNLI_labelmap, 'bad info found in {}'.format(caption_file)
            self.data.append([info['pairID'], info['sentence2'], info['gold_label'], info['Flikr30kID']])

        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def __getitem__(self, index):
        question_id, question, label, image_id = self.data[index]
        feature = self.get_image(image_id)
        label_id = VNLI_labelmap[label]
        return index, (feature, question, label_id)

    def get_question_id(self, index):
        return self.data[index][0]

    def get_image(self, image_id):
        if image_id.endswith('.jpg'):
            img_path = os.path.join(self.image_dir, image_id)
        else:
            img_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
        img = Image.open(img_path).convert('RGB')
        return self.img_transform(img)

    def __len__(self):        
        return len(self.data)


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels 
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores



def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            # model_to_save.save_pretrained(checkpoint_dir)
            object_to_save = {
                            'model': model_to_save.state_dict(),
                            'step': global_step
                        }
            torch.save(object_to_save, os.path.join(checkpoint_dir,
                                                'ckpt.pth'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    opt_arg = albef_utils.AttrDict(args.opt['optimizer'])
    optimizer = create_optimizer(opt_arg, model)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()
    set_seed(args.seed, args.n_gpu)
    log_json = []
    best_score = 0
    if args.time_debug:
        # record the training time
        global_data_time = 0.0
        global_compute_time = 0.0
        time_start = time.time()
    for epoch in range(int(args.num_train_epochs)):
        t_start = time.time()
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            images, text, answer = batch
            if global_step == 0:
                print(text, answer)
            text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
            answer = answer.to(args.device)
            if args.time_debug:
                time_point1 = time.time()
            images = images.to(args.device)
            bs = images.shape[0]
            # print('before', text.input_ids.shape)
            outputs = model(images, text.input_ids, text.attention_mask, answer)
            loss, logits = outputs
            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.time_debug:
                time_point2 = time.time()
            # pseudo_labels = torch.cat([torch.ones(sim_mat.shape[0]), torch.zeros(bs)], dim=0).to(dtype=torch.long, device=logits.device)
            batch_score = compute_score_with_logits(logits, answer).sum()
            batch_acc = batch_score.item() / (args.train_batch_size) # multipled by 3 since 1 pos and 3 negative sample
            global_loss += loss.item()
            global_acc += batch_acc
            if args.time_debug:
                data_time = time_point1 - time_start
                compute_time = time_point2-time_point1
                global_data_time += data_time
                global_compute_time += compute_time
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                # if global_step % 100 == 0:
                #     eval_result, eval_score = evaluate(args, model, val_dataset, tokenizer)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / (global_step * args.gradient_accumulation_steps), 
                        batch_acc, global_acc / (global_step * args.gradient_accumulation_steps))
                    )
                    if args.time_debug:
                        logger.info('time info: data_time: {:.4f} ({:.4f}), compute time: {:.4f} ({:.4f})'.format(data_time, global_data_time / (global_step*args.gradient_accumulation_steps), compute_time, global_compute_time / (global_step*args.gradient_accumulation_steps)))

            if args.time_debug:
                time_start = time.time()

        if args.evaluate_during_training:
            # do evaluation
            logger.info("Perform evaluation at step: %d" % (global_step))
            # only VSE retrieval
            if isinstance(val_dataset, dict):
                eval_score = 0
                full_score = {}
                for k,v in val_dataset.items():
                    logger.info('evaluation on language {}'.format(k))
                    tmp_res, tmp_score = evaluate(args, model, v, tokenizer)
                    full_score[k] = tmp_score
                    eval_score += tmp_score
            else:
                eval_result, eval_score = evaluate(args, model, val_dataset, tokenizer)

            current_score = eval_score
            if current_score > best_score:
                best_score = current_score
            # if rank_accs['R@1'] > best_score:
            #     best_score = rank_accs['R@1']
            epoch_log = {'epoch': epoch, 'global_step': global_step, 'current_score': current_score,
                            'best_score':best_score}
            if isinstance(val_dataset, dict):
                epoch_log['full_score'] = full_score
            log_json.append(epoch_log)
            if args.local_rank in [-1, 0]:
                with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                    json.dump(log_json, fp) 

            logger.info("EVALACC: {}%".format(100*best_score))

        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch, t_end - t_start))

        if args.local_rank in [-1, 0] and ((args.save_epoch > 0 and epoch % args.save_epoch == 0) or \
                global_step == t_total):
            save_checkpoint(model, tokenizer, args, epoch, global_step) 
            
    return global_step, global_loss / global_step

def prepare_inputs(inputs, args):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if inputs[k].dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                inputs[k]=v.to(dtype=args.dtype)
    return inputs

def test(args, model, eval_dataset, tokenizer):
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    label2ans = {v:k for k,v in VNLI_labelmap.items()}
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    num_data = 0
    score = 0.0
    upper_bound = 0
    results_dict = {}
    t_start = time.time()

    for index, batch in tqdm(eval_dataloader, desc='testing'):
    #for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        images, text, answer = batch
        text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
        answer = answer.to(args.device)
        if args.time_debug:
            time_point1 = time.time()
        images = images.to(args.device)
        bs = images.shape[0]
        # print('before', text.input_ids.shape)

        with torch.no_grad():
            outputs = model(images, text.input_ids, text.attention_mask, answer)

            tmp_eval_loss, logits = outputs[:2]
            preds = torch.max(logits, 1)[1].data

            eval_loss += tmp_eval_loss.mean().item()

            # batch_score = compute_score_with_logits(logits, batch[4]).sum()
            batch_score = compute_score_with_logits(logits, answer)
            # update results_dict
            results_dict.update(
                {eval_dataset.get_question_id(qa_ind): label2ans[p] for qa_ind, p in
                    zip(index, preds.tolist())}
            )
            score += batch_score.sum().item()
            #upper_bound += (batch[4].max(1)[0]).sum().item()
            num_data += logits.size(0)

            # debug
            #val, idx = logits.max(1)
            #logger.info('idx: %s, batch[4]: %s' % (str(idx.shape), str(batch[3].shape)))
            #for i in range(idx.size(0)):
            #    logger.info('idx: %d, pred: %d, real: %d' % (idx[i].item(), eval_dataset.labels[idx[i].item()], batch[3][i].item()))

        nb_eval_steps += 1

        #if preds is None:
        #    preds = logits.detach().cpu().numpy()
        #    out_label_ids = inputs['labels'].detach().cpu().numpy()
        #else:
        #    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    score = score / len(eval_dataloader.dataset)

    logger.info("Eval Results:")
    logger.info("Eval Error: %.3f" % (eval_loss/nb_eval_steps))
    logger.info("Eval Score: {}%".format(100*score))
    # with open(os.path.join(args.data_dir, 'val_results.json'),
    #           'w') as f:
    #     json.dump(results_dict, f)

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))

        #eval_loss = eval_loss / nb_eval_steps
        #if args.output_mode == "classification":
        #    preds = np.argmax(preds, axis=1)
        #elif args.output_mode == "regression":
        #    preds = np.squeeze(preds)
        #result = compute_metrics(eval_task, preds, out_label_ids)
        #results.update(result)

        #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        #with open(output_eval_file, "w") as writer:
        #    logger.info("***** Eval results {} *****".format(prefix))
        #    for key in sorted(result.keys()):
        #        logger.info("  %s = %s", key, str(result[key]))
        #        writer.write("%s = %s\n" % (key, str(result[key])))

    return results_dict, score



def evaluate(args, model, eval_dataset=None, tokenizer=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    #if args.n_gpu > 1: model = torch.nn.DataParallel(model) # debug: single-gpu or multi-gpus

    results = []
    t_start = time.time()

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, num_workers=args.num_workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    num_data = 0
    score = 0.0
    upper_bound = 0
    results_dict = {}
    tmp_score = 0.0

    for index, batch in tqdm(eval_dataloader, desc='evaluation'):
    #for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        images, text, answer = batch
        text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
        answer = answer.to(args.device)
        if args.time_debug:
            time_point1 = time.time()
        images = images.to(args.device)
        bs = images.shape[0]
        # print('before', text.input_ids.shape)

        with torch.no_grad():
            outputs = model(images, text.input_ids, text.attention_mask, answer)

            tmp_eval_loss, logits = outputs[:2]
            preds = torch.max(logits, 1)[1].data

            eval_loss += tmp_eval_loss.mean().item()

            # batch_score = compute_score_with_logits(logits, batch[4]).sum()
            batch_score = compute_score_with_logits(logits, answer)
            # update results_dict
            results_dict.update(
                {eval_dataset.get_question_id(qa_ind): p for qa_ind, p in
                    zip(index, preds.tolist())}
            )
            tmp_score += batch_score.float().mean().item()
            score += batch_score.sum().item()
            #upper_bound += (batch[4].max(1)[0]).sum().item()
            num_data += logits.size(0)

            # debug
            #val, idx = logits.max(1)
            #logger.info('idx: %s, batch[4]: %s' % (str(idx.shape), str(batch[3].shape)))
            #for i in range(idx.size(0)):
            #    logger.info('idx: %d, pred: %d, real: %d' % (idx[i].item(), eval_dataset.labels[idx[i].item()], batch[3][i].item()))

        nb_eval_steps += 1

        if nb_eval_steps % 200 == 0:
            logger.info('current score: {}'.format(tmp_score / (nb_eval_steps)))
            with open('/remote-home/zjli/tmp_debug/tmp_res.json', 'w') as wf:
                json.dump(results_dict, wf)

        #if preds is None:
        #    preds = logits.detach().cpu().numpy()
        #    out_label_ids = inputs['labels'].detach().cpu().numpy()
        #else:
        #    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    score = score / len(eval_dataloader.dataset)

    logger.info("Eval Results:")
    logger.info("Eval Loss: %.3f" % (eval_loss/nb_eval_steps))
    logger.info("Eval Score: {}%".format(100*score))
    # with open(os.path.join(args.data_dir, 'val_results.json'),
    #           'w') as f:
    #     json.dump(results_dict, f)

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))

        #eval_loss = eval_loss / nb_eval_steps
        #if args.output_mode == "classification":
        #    preds = np.argmax(preds, axis=1)
        #elif args.output_mode == "regression":
        #    preds = np.squeeze(preds)
        #result = compute_metrics(eval_task, preds, out_label_ids)
        #results.update(result)

        #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        #with open(output_eval_file, "w") as writer:
        #    logger.info("***** Eval results {} *****".format(prefix))
        #    for key in sorted(result.keys()):
        #        logger.info("  %s = %s", key, str(result[key]))
        #        writer.write("%s = %s\n" % (key, str(result[key])))

    return results_dict, score


def get_predict_file(args):
    cc = []
    data = op.basename(op.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    return op.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc))) 


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
            'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def reinit_qa_head(tokenizer, ans2label, weight, bias):
    label2ans = {v:k for k,v in ans2label.items()}
    label2ans[len(label2ans)] = 'none'
    new_weight = []
    new_bias = []
    with torch.no_grad():
        for i in tqdm(range(len(label2ans)), desc='re-initializing the qa head weight'):
            c_ans = label2ans[i]
            c_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(c_ans))
            c_weight = torch.mean(weight[c_ids], dim=0)
            c_bias = torch.mean(bias[c_ids], dim=0)
            new_weight.append(c_weight)
            new_bias.append(c_bias)
    new_weight = torch.stack(new_weight)
    new_bias = torch.stack(new_bias)
    return new_weight, new_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False,
                        help="The input data dir with all required files.")
    # parser.add_argument("--img_dir", default='datasets/coco_ir/features.tsv', type=str, required=False,
    #                     help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset" 
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                       "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_epoch', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument('--extra_concept', action='store_true', help="Whether to add more related concepts from the concept graph.")
    parser.add_argument('--num_extra_concept', type=int, default=5, help="Number of extra concapts added")
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help="Which GPUs to use")
    parser.add_argument('--half_evaluation', action='store_true', help='Whther to use half precision for evaluation')
    parser.add_argument('--dataset_name', type=str, default='flickr', help='which dataset is using')
    parser.add_argument('--max_tag_length', type=int, default=20)
    parser.add_argument('--text_tokenizer', type=str, default=None)
    parser.add_argument('--albef_config', type=str, default=None)
    parser.add_argument('--eval_all_checkpoints', action='store_true')
    parser.add_argument('--eval_split', type=str, default='val')
    parser.add_argument('--time_debug', action='store_true', help='print time info for debugging')
    parser.add_argument('--cuda_devices', type=str, default=None, help='sub-part of cuda devices')
    parser.add_argument('--model_type', type=str, default=None, help='explicitly indicate the model type')
    parser.add_argument('--ans2label_map', type=str, default=None, help='the answer to label map')
    parser.add_argument('--original_tokenizer', type=str, default=None, help='used to recover from the original ckpt after stage1')
    parser.add_argument('--original_ckpt', type=str, default=None, help='the ckpt file for above')
    parser.add_argument('--test_lang', type=str, default='en', help='the target test language')
    parser.add_argument('--train_by_image', action='store_true', help='use image dimension as dataset')
    parser.add_argument('--image_dir_format', type=str, default=None, help='the image dir format, local or remote-home')
    # parser.add_argument('--data_file_train', type=str, default=None, help='the caption data file of training')
    # parser.add_argument('--data_file_val', type=str, default=None, help='the caption data file of validation')
    args = parser.parse_args()

    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        
    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # set_seed(args.seed, args.n_gpu)
    # logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    # logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, AutoTokenizer
    albef_config = yaml.load(open(args.albef_config, 'r'), Loader=yaml.Loader)
    bert_config = BertConfig.from_json_file(albef_config['bert_config'])

    # dataset loading
    args.image_dir = albef_config['image_root'][args.image_dir_format]
    args.data_file_train = albef_config['train_file']
    if args.eval_split == 'val':
        if args.test_lang == 'all':
            args.data_file_val = albef_config['val_file']
        else:
            args.data_file_val = albef_config['val_file'][args.text_lang]
    elif args.eval_split == 'test':
        if args.test_lang == 'all':
            args.data_file_val = albef_config['test_file']
        else:
            args.data_file_val = albef_config['test_file'][args.test_lang]
    elif args.eval_split == 'translate_test':
        if args.test_lang == 'all':
            args.data_file_val = albef_config['translate_test_file']
        else:
            args.data_file_val = albef_config['translate_test_file'][args.test_lang]
    else:
        raise NotImplementedError

    if args.model_type is not None:
        albef_config['model_type'] = args.model_type
    else:
        args.model_type = albef_config['model_type']
    
    if albef_config['model_type'] == 'classification': 
        model_class = ALBEFforClassification
    else:
        raise NotImplementedError

    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path)
        model = model_class(config=albef_config, text_encoder=albef_config['text_encoder'], tokenizer=tokenizer)
        args.dtype = torch.float32
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        try:
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
        except:
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class(config=albef_config, text_encoder=albef_config['text_encoder'], tokenizer=tokenizer)
        args.model_name_or_path = checkpoint
        # model = model_class.from_pretrained(checkpoint, config=config)
        if args.half_evaluation:
            model = model.half()
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32
    
    if args.model_name_or_path is not None:
        if args.model_name_or_path.endswith('.pth'):
            ckpt_file = args.model_name_or_path
        else:
            ckpt_file = os.path.join(args.model_name_or_path, 'ckpt.pth')
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        logger.info(" -> Recovering model checkpoint from {}".format(ckpt_file))
        target_keys = set(model.state_dict().keys())
        # tmp_keys = ['text_encoder.bert.embeddings.word_embeddings.weight', 'text_encoder.cls.predictions.decoder.bias', 'text_encoder.cls.predictions.decoder.weight', 'text_encoder.cls.predictions.bias']
        tmp_keys = []
        state_dict = {k:v.to(torch.float32) if v.dtype==torch.float16 else v for k,v in checkpoint['model'].items()} # if k in target_keys and k not in tmp_keys}

        if model.visual_encoder.pos_embed.shape[1] != state_dict['visual_encoder.pos_embed'].shape[1]:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)     
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if args.do_train:
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.','')         
                    state_dict[encoder_key] = state_dict[key] 
                if args.model_type == 'classification':
                    if 'text_encoder' in key:
                        del state_dict[key]

            if model.text_encoder.embeddings.word_embeddings.weight.shape[0] != state_dict['text_encoder.embeddings.word_embeddings.weight'].shape[0]:
                # re-initialize the embedding weight and prediction head for embeddings of a new size
                target_size = model.text_encoder.embeddings.word_embeddings.weight.shape
                new_embs = torch.randn(target_size)*bert_config.initializer_range
                state_dict['text_encoder.embeddings.word_embeddings.weight'] = new_embs
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)

    freeze_part = albef_config['freeze_module'] if 'freeze_module' in albef_config else []
    for module_name in freeze_part:
        logger.info('freezing the {} module'.format(module_name))
        model.freeze(module_name)
    
    model.to(args.device)
    train_transform = create_transform(config=albef_config, name='train')
    val_transform = create_transform(config=albef_config, name='test')
    logger.info("Training/evaluation parameters %s", args)
    args.opt = albef_config
    if args.do_train:
        train_dataset = xVNLIDataset(caption_file=args.data_file_train, img_transform=train_transform, tokenizer=tokenizer, args=args, is_train=True)
        if args.evaluate_during_training:
            if isinstance(args.data_file_val, dict):
                val_dataset = {k: xVNLIDataset(caption_file=v, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False) for k,v in args.data_file_val.items()}
            else:
                val_dataset = xVNLIDataset(caption_file=args.data_file_val, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False)
        else:
            val_dataset = None

        # test_res, test_score = test(args, model, val_dataset, tokenizer=tokenizer)
        # return None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            # args = restore_training_settings(args)
            test_dataset = xVNLIDataset(caption_file=args.data_file_val, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False)
            for checkpoint in checkpoints:
                # if checkpoint.split('-')[-2] <= '10':
                #     continue
                assert op.isdir(checkpoint)
                logger.info("Evaluate the following checkpoint: %s", checkpoint)
                model = model_class.from_pretrained(checkpoint, config=config)
                if args.half_evaluation:
                    model = model.half()
                    args.dtype = torch.float16
                else:
                    args.dtype = torch.float32
                model.to(args.device)
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                #pred_file = get_predict_file(args)
                # if op.isfile(pred_file):
                #     logger.info("Prediction file exist, skip inference.")
                #     if args.do_eval:
                #         test_result = torch.load(pred_file)
                # else:
                #     test_result = test(args, model, test_dataset)
                #     torch.save(test_result, pred_file)
                #     logger.info("Prediction results saved to {}.".format(pred_file))

                coarse_sim = test_coarse(args, model, test_dataset, tokenizer=tokenizer)
                eval_result, caption_index, image_index = evaluate_coarse(test_dataset, coarse_sim)
                # caption index and image index
                eval_i2t_result, _ = test_fine_i2t(args, model, test_dataset, caption_index=caption_index, tokenizer=tokenizer)
                eval_t2i_result = test_fine_t2i(args, model, test_dataset, image_index=image_index, tokenizer=tokenizer)
                print('fine inference:')
                # print(eval_i2t_result, eval_t2i_result)
                if args.do_eval:
                    eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)
                    # result_file = op.splitext(pred_file)[0] + '.eval.json'
                    result_file = op.join(checkpoint, 'test_eval.json')
                    with open(result_file, 'w') as f:
                        json.dump(eval_result, f)
                    logger.info("Evaluation results saved to {}.".format(result_file))
        else:
            # args = restore_training_settings(args)
            if isinstance(args.data_file_val, dict):
                test_dataset = {k: xVNLIDataset(caption_file=v, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False) for k,v in args.data_file_val.items()}
            else:
                test_dataset = xVNLIDataset(caption_file=args.data_file_val, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False)
            checkpoint = args.eval_model_dir
            assert op.isdir(checkpoint)
            logger.info("Evaluate the following checkpoint: %s", checkpoint)

            if args.half_evaluation:
                model = model.half()
                args.dtype = torch.float16
            else:
                args.dtype = torch.float32
            model.to(args.device)
            # if args.n_gpu > 1:
            #     model = torch.nn.DataParallel(model)

            pred_file = get_predict_file(args)
            # if op.isfile(pred_file):
            #     logger.info("Prediction file exist, skip inference.")
            #     if args.do_eval:
            #         test_result = torch.load(pred_file)
            # else:
            #     test_result = test(args, model, test_dataset)
            #     torch.save(test_result, pred_file)
            #     logger.info("Prediction results saved to {}.".format(pred_file))
            if isinstance(test_dataset, dict):
                eval_score = 0
                full_score = {}
                for k,v in test_dataset.items():
                    logger.info('evaluation on language {}'.format(k))
                    tmp_res, tmp_score = test(args, model, v, tokenizer)
                    full_score[k] = tmp_score
                    eval_score += tmp_score
            else:
                logger.info('evaluation on language {}'.format(args.test_lang))
                eval_result, eval_score = test(args, model, test_dataset, tokenizer)

if __name__ == "__main__":
    main()
