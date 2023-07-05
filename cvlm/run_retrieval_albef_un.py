# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
from __future__ import absolute_import, division, print_function
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import base64
import os.path as op
import random, json
from ssl import _create_unverified_context
import numpy as np
import torch
import glob
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from tqdm import tqdm
import yaml
from albef import utils as albef_utils
import time

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed, weighted_sample
from albef.modeling.model_retrieval import ALBEF
from transformers_past.pytorch_transformers import BertTokenizer, BertConfig 
from transformers_past.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from transformers_past.pytorch_transformers.modeling_utils import WEIGHTS_NAME
from PIL import Image
from transformers import AutoTokenizer
from albef.modeling.vit import interpolate_pos_embed
from albef.modeling.transforms_utils import create_transform
from albef.optim import create_optimizer

WEIGHTS_NAME = 'ckpt.pth'

class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    # unbalanced version for retrieval
    def __init__(self, caption_file, img_transform, tokenizer, args, is_train=True, coarse_cap_index=None, coarse_img_index=None):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(RetrievalDataset, self).__init__()
        self.image_dir = args.image_dir
        self.img_transform = img_transform
        if caption_file.endswith('.pth') or caption_file.endswith('.pt'):
            # pytorch file
            self.captions = torch.load(caption_file)
        elif caption_file.endswith('.json'):
            # json file
            self.captions = json.load(open(caption_file, 'r'))
        elif caption_file.endswith('jsonl'):
            # jsonl file
            self.captions = {}
            for line in open(caption_file, 'r'):
                info = json.loads(line)
                self.captions[info['img_path']] = info['sentences']
        else:
            raise NotImplementedError
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        self.complete_captions = []
        self.cap_id2img_id = {}
        self.img_keys = []
        self.num_of_total_captions = 0
        
        for img_index, (k,v) in tqdm(enumerate(self.captions.items())):
            self.img_keys.append(k)
            for i,c in enumerate(v):
                self.cap_id2img_id[self.num_of_total_captions] = img_index
                self.complete_captions.append(c)
                self.num_of_total_captions += 1

        # self.num_of_total_captions = args.num_captions_per_img_train*len(self.img_keys)
        print('number of total captions:',self.num_of_total_captions)
        print('number of images', len(self.img_keys))

        # get the image image_id to index map
        # imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        # self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        
        # get the image features and label


        self.ds_name = args.dataset_name
        # self.img2theme = {k:v for k,v in self.img2theme.items() if k.startswith(self.ds_name)}

        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            self.num_images_per_cap = args.num_images_per_cap_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.num_of_total_captions = args.num_captions_per_img_train*len(self.img_keys)
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}

            if args.eval_caption_index_file:
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval 
                caption_index_file = op.join(args.data_dir, args.eval_caption_index_file)
                self.caption_indexs = torch.load(caption_index_file)
                if not type(self.caption_indexs[self.img_keys[0]]) == list:
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
            else:
                self.has_caption_indexs = False

            if coarse_cap_index:
                self.has_caption_indexs = True
                self.caption_indexs = coarse_cap_index
            else:
                self.has_caption_indexs = False

            if coarse_img_index:
                self.has_image_indexs = True
                self.image_indexs = coarse_img_index
            else:
                self.has_image_indexs = False

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def set_caption_index(self, caption_index):
        self.num_captions_per_img = self.args.num_captions_per_img_val
        self.has_caption_indexs = True
        self.has_image_indexs = False
        self.caption_indexs = caption_index

    def set_image_index(self, image_index):
        self.num_images_per_cap = self.args.num_images_per_cap_val
        self.has_image_indexs = True
        self.has_caption_indexs = False
        self.image_indexs = image_index

    def unset_index(self):
        self.num_captions_per_img = self.args.num_captions_per_img_train
        self.num_images_per_cap = 1
        self.has_image_indexs = False
        self.has_caption_indexs = False
    
    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_of_total_captions)
            cap_idx = index % (self.num_of_total_captions)
            return img_idx, cap_idx
        if not self.is_train and self.has_caption_indexs:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            cap_idx_target = self.caption_indexs[img_idx][cap_idx]
            return img_idx, cap_idx_target
        if not self.is_train and self.has_image_indexs:
            cap_idx = index // self.num_images_per_cap
            img_idx = index % self.num_images_per_cap
            img_key1 = self.image_indexs[cap_idx][img_idx]
            return img_key1, cap_idx
        img_idx = self.cap_id2img_id[index]
        cap_idx = index
        return img_idx, cap_idx

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.cap_id2img_id[cap_idx]==img_idx else 0

    def get_label_fine(self, img_index, cap_index):
        return 1 if self.cap_id2img_id[cap_index]==img_index else 0

    def __getitem__(self, index):
        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        feature = self.get_image(img_key)
        caption = self.complete_captions[cap_idxs]
        # caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        label = 1 if self.cap_id2img_id[cap_idxs]==img_idx else 0
        # print([i.shape for i in example])
        return index, (feature, caption, label)

    def get_image(self, image_full_id):
        if 'COCO' in image_full_id:
            # coco with split name
            split = image_full_id.split('_')[1]
            image_id = os.path.join(split, image_full_id)
        else:
            image_id = image_full_id
        if image_id.endswith('.jpg'):
            img_path = os.path.join(self.image_dir, image_id)
        else:
            img_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
        img = Image.open(img_path).convert('RGB')
        return self.img_transform(img)
        # theme_nodes = self.img2theme[self.ds_name+'_'+str(image_id)]
        # if len(theme_nodes) > self.args.max_visual_themes:
        #     theme_nodes = theme_nodes[:self.args.max_visual_themes]
        # theme_nodes = [t[0]+self.tokenizer.vocab_size+self.phrase_vocab_size for t in theme_nodes]
        # return t_features, theme_nodes

    def get_img_sub_index(self):
        # return the sub image index list for indexing
        sub_indexes = []
        last = None
        for i in range(self.num_of_total_captions):
            img_id = self.cap_id2img_id[i]
            if img_id != last:
                sub_indexes.append(i)
                last = img_id
        assert len(self.img_keys)==len(sub_indexes)
        return sub_indexes

    def __len__(self):
        if self.is_train:
            return self.num_of_total_captions
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) * self.num_of_total_captions
        if not self.is_train and self.has_image_indexs:
            return self.num_images_per_cap * self.num_of_total_captions
        if not self.is_train and self.has_caption_indexs:
            return len(self.img_keys) * self.num_captions_per_img
        return self.num_of_total_captions


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


def compute_ranks(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_of_total_captions
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])

    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def compute_ranks_t2i(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    assert dataset.has_image_indexs
    num_images_per_cap = dataset.num_images_per_cap
    labels = np.reshape(labels, [-1, num_images_per_cap])
    similarities = np.reshape(similarities, [-1, num_images_per_cap])
    t2i_ranks = []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_images_per_cap
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
    return t2i_ranks


def compute_ranks_coarse(dataset, similarities):
    i2t_ranks, t2i_ranks = [], []
    i2t_index = {}
    t2i_index = {}
    # i2t
    for i in range(similarities.shape[0]):
        tmp_index = []
        inds = np.argsort(similarities[i,:])[::-1]
        rank = similarities.shape[1]
        for r, ind in enumerate(inds):
            # if ind >= i*dataset.args.num_captions_per_img_train and ind < (i+1)*dataset.args.num_captions_per_img_train:
            if dataset.get_label_fine(i, ind):
                rank = r
                break
        i2t_ranks.append(rank)
        for r, ind in enumerate(inds):
            if r >= dataset.args.num_captions_per_img_val:
                break
            # cap_img_index = ind // dataset.args.num_captions_per_img_train
            # cap_cap_index = ind % dataset.args.num_captions_per_img_train
            tmp_index.append(ind)

        i2t_index[i] = tmp_index

    # t2i
    for i in range(similarities.shape[1]):
        tmp_index = []
        inds = np.argsort(similarities[:,i])[::-1]
        rank = similarities.shape[0]
        for r, ind in enumerate(inds):
            # if ind == i//dataset.args.num_captions_per_img_train:
            if dataset.get_label_fine(ind, i):
                rank = r
                break
        t2i_ranks.append(rank)
        for r, ind in enumerate(inds):
            if r >= dataset.args.num_images_per_cap_val:
                break
            tmp_index.append(ind)

        t2i_index[i] = tmp_index
        # t2i_index[(dataset.img_keys[cap_img_index], cap_cap_index)] = tmp_index
    return i2t_ranks, t2i_ranks, i2t_index, t2i_index


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
    if (len(train_dataset) % args.train_batch_size) < args.n_gpu*2:
        dp_last = True
        logger.info('founded unsupported dataset size, using drop last!')
    else:
        dp_last = False
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers, drop_last=dp_last)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    opt_arg = albef_utils.AttrDict(args.opt['optimizer'])
    if args.learning_rate is not None:
        opt_arg['lr'] = args.learning_rate
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

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    global_r_loss, global_f_loss = 0.0, 0.0
    model.zero_grad()
    log_json = []
    best_score = 0
    if args.time_debug:
        # record the training time
        global_data_time = 0.0
        global_compute_time = 0.0
        time_start = time.time()
    
    if args.evaluate_before_training:
        logger.info("Perform evaluation at step: %d" % (global_step))
        if isinstance(val_dataset, dict):
            # only VSE retrieval
            current_score = 0
            full_lang_res = {}
            for k,v in val_dataset.items():
                logger.info("Evaluation on language {}".format(k))
                coarse_sim = test_coarse(args, model, v, tokenizer)
                eval_result, caption_index, image_index = evaluate_coarse(v, coarse_sim)
                # caption index and image index
                eval_i2t_result, _ = test_fine_i2t(args, model, v, caption_index=caption_index, tokenizer=tokenizer)
                eval_t2i_result = test_fine_t2i(args, model, v, image_index=image_index, tokenizer=tokenizer)
                print('fine inference:')
                # print(eval_i2t_result, eval_t2i_result)
                eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)

                full_lang_res[k] = {'I2T': eval_result['i2t_retrieval'], 'T2I': eval_result['t2i_retrieval']}
                rank_accs = eval_result['i2t_retrieval']
                if args.save_metric == 'r1':
                    current_score += rank_accs['R@1']
                elif args.save_metric == 'rsum':
                    current_score += sum(eval_result['i2t_retrieval'].values()) + sum(eval_result['t2i_retrieval'].values())
                elif args.save_metric == 'mR':
                    current_full_scores = list(eval_result['i2t_retrieval'].values()) + list(eval_result['t2i_retrieval'].values())
                    current_score += sum(current_full_scores) / (len(current_full_scores))
            if current_score > best_score:
                best_score = current_score
            # if rank_accs['R@1'] > best_score:
            #     best_score = rank_accs['R@1']
            epoch_log = {'epoch': 0, 'global_step': 0, 
                        'recall': full_lang_res, 'best_{}'.format(args.save_metric):best_score}
        else:
            # only VSE retrieval
            coarse_sim = test_coarse(args, model, val_dataset, tokenizer)
            eval_result, caption_index, image_index = evaluate_coarse(val_dataset, coarse_sim)
            # caption index and image index
            eval_i2t_result, _ = test_fine_i2t(args, model, val_dataset, caption_index=caption_index, tokenizer=tokenizer)
            eval_t2i_result = test_fine_t2i(args, model, val_dataset, image_index=image_index, tokenizer=tokenizer)
            print('fine inference:')
            # print(eval_i2t_result, eval_t2i_result)
            eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)

            rank_accs = eval_result['i2t_retrieval']
            if args.save_metric == 'r1':
                current_score = rank_accs['R@1']
            elif args.save_metric == 'rsum':
                current_score = sum(eval_result['i2t_retrieval'].values()) + sum(eval_result['t2i_retrieval'].values())
            elif args.save_metric == 'mR':
                current_full_scores = list(eval_result['i2t_retrieval'].values()) + list(eval_result['t2i_retrieval'].values())
                current_score = sum(current_full_scores) / (len(current_full_scores))
            if current_score > best_score:
                best_score = current_score
            # if rank_accs['R@1'] > best_score:
            #     best_score = rank_accs['R@1']
            epoch_log = {'epoch': 0, 'global_step': 0, 
                        'R1': rank_accs['R@1'], 'R5': rank_accs['R@5'], 
                        'R10': rank_accs['R@10'], 'best_{}'.format(args.save_metric):best_score}
        log_json.append(epoch_log)
        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp) 

    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            if hasattr(model, 'module'):
                model.module.forward_mod = 'train'
            else:
                model.forward_mod = 'train'
            images, text, labels = batch
            text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
            if args.time_debug:
                time_point1 = time.time()
            images = images.to(args.device)
            bs = images.shape[0]
            # print('before', text.input_ids.shape)
            outputs = model(images, text.input_ids, text.attention_mask)
            loss, logits, r_loss, f_loss, pseudo_labels = outputs
            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
                r_loss = r_loss.mean()
                f_loss = f_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.time_debug:
                time_point2 = time.time()
            # pseudo_labels = torch.cat([torch.ones(sim_mat.shape[0]), torch.zeros(bs)], dim=0).to(dtype=torch.long, device=logits.device)
            batch_score = compute_score_with_logits(logits, pseudo_labels).sum()
            batch_acc = batch_score.item() / (args.train_batch_size * 3) # multipled by 3 since 1 pos and 3 negative sample
            global_loss += loss.item()
            global_r_loss += r_loss.item()
            global_f_loss += f_loss.item()
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
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "CLIP_loss: {:.4f} ({:.4f}), HN_loss: {:.4f} ({:.4f}), score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        r_loss.item(), global_r_loss/global_step, f_loss.item(), global_f_loss/global_step, batch_acc, global_acc / global_step)
                    )
                    if args.time_debug:
                        logger.info('time info: data_time: {:.4f} ({:.4f}), compute time: {:.4f} ({:.4f})'.format(data_time, global_data_time / global_step, compute_time, global_compute_time / global_step))

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        if isinstance(val_dataset, dict):
                            # only VSE retrieval
                            current_score = 0
                            full_lang_res = {}
                            for k,v in val_dataset.items():
                                logger.info("Evaluation on language {}".format(k))
                                coarse_sim = test_coarse(args, model, v, tokenizer)
                                eval_result, caption_index, image_index = evaluate_coarse(v, coarse_sim)
                                # caption index and image index
                                eval_i2t_result, _ = test_fine_i2t(args, model, v, caption_index=caption_index, tokenizer=tokenizer)
                                eval_t2i_result = test_fine_t2i(args, model, v, image_index=image_index, tokenizer=tokenizer)
                                print('fine inference:')
                                # print(eval_i2t_result, eval_t2i_result)
                                eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)

                                full_lang_res[k] = {'I2T': eval_result['i2t_retrieval'], 'T2I': eval_result['t2i_retrieval']}
                                rank_accs = eval_result['i2t_retrieval']
                                if args.save_metric == 'r1':
                                    current_score += rank_accs['R@1']
                                elif args.save_metric == 'rsum':
                                    current_score += sum(eval_result['i2t_retrieval'].values()) + sum(eval_result['t2i_retrieval'].values())
                                elif args.save_metric == 'mR':
                                    current_full_scores = list(eval_result['i2t_retrieval'].values()) + list(eval_result['t2i_retrieval'].values())
                                    current_score += sum(current_full_scores) / (len(current_full_scores))

                            if current_score > best_score:
                                best_score = current_score
                            # if rank_accs['R@1'] > best_score:
                            #     best_score = rank_accs['R@1']
                            epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                        'recall': full_lang_res, 'best_{}'.format(args.save_metric):best_score}
                        else:
                            # only VSE retrieval
                            coarse_sim = test_coarse(args, model, val_dataset, tokenizer)
                            eval_result, caption_index, image_index = evaluate_coarse(val_dataset, coarse_sim)
                            # caption index and image index
                            eval_i2t_result, _ = test_fine_i2t(args, model, val_dataset, caption_index=caption_index, tokenizer=tokenizer)
                            eval_t2i_result = test_fine_t2i(args, model, val_dataset, image_index=image_index, tokenizer=tokenizer)
                            print('fine inference:')
                            # print(eval_i2t_result, eval_t2i_result)
                            eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)

                            rank_accs = eval_result['i2t_retrieval']
                            if args.save_metric == 'r1':
                                current_score = rank_accs['R@1']
                            elif args.save_metric == 'rsum':
                                current_score = sum(eval_result['i2t_retrieval'].values()) + sum(eval_result['t2i_retrieval'].values())
                            elif args.save_metric == 'mR':
                                current_full_scores = list(eval_result['i2t_retrieval'].values()) + list(eval_result['t2i_retrieval'].values())
                                current_score = sum(current_full_scores) / (len(current_full_scores))
                            if current_score > best_score:
                                best_score = current_score
                            # if rank_accs['R@1'] > best_score:
                            #     best_score = rank_accs['R@1']
                            epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                        'R1': rank_accs['R@1'], 'R5': rank_accs['R@5'], 
                                        'R10': rank_accs['R@10'], 'best_{}'.format(args.save_metric):best_score}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 
            if args.time_debug:
                time_start = time.time()
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

def test_coarse(args, model, eval_dataset, tokenizer):
    # 2 stage evaluation
    if hasattr(model, 'module'):
        model.module.forward_mod = 'coarse'
    else:
        model.forward_mod = 'coarse'
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset.unset_index()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    full_txt_emb = []
    full_img_emb = []
    for indexs, batch in tqdm(eval_dataloader):
        with torch.no_grad():
            images, text, labels = batch
            text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
            images = images.to(args.device)
            bs = images.shape[0]
            global_txt, global_img = model(images, text.input_ids, text.attention_mask)[:2]
            full_txt_emb.append(global_txt)
            full_img_emb.append(global_img)
    with torch.no_grad():
        full_txt_emb = torch.cat(full_txt_emb, dim=0)
        full_img_emb = torch.cat(full_img_emb, dim=0)
        select_index = eval_dataset.get_img_sub_index()
        # num_imgs = int(full_img_emb.shape[0] / args.num_captions_per_img_train)
        # assert(full_img_emb.shape[0] % args.num_captions_per_img_train == 0)
        # select_index = [i*args.num_captions_per_img_train for i in range(num_imgs)]
        full_img_emb = full_img_emb[select_index]
        full_sims = full_img_emb @ full_txt_emb.t()
        print(full_sims.shape)
    return full_sims.detach().cpu().numpy()

def test_fine_t2i(args, model, eval_dataset, image_index, tokenizer):
    # 2 stage evaluation
    if hasattr(model, 'module'):
        model.module.forward_mod = 'fine'
    else:
        model.forward_mod = 'fine'
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset.set_image_index(image_index)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        with torch.no_grad():
            images, text, labels = batch
            text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
            images = images.to(args.device)
            bs = images.shape[0]
            logits = model(images, text.input_ids, text.attention_mask)
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return compute_ranks_t2i(eval_dataset, results)


def test_fine_i2t(args, model, eval_dataset, caption_index, tokenizer):
    # 2 stage evaluation
    if hasattr(model, 'module'):
        model.module.forward_mod = 'fine'
    else:
        model.forward_mod = 'fine'
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset.set_caption_index(caption_index)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        with torch.no_grad():
            images, text, labels = batch
            text = tokenizer(list(text), padding='longest', max_length=args.max_seq_length, return_tensors="pt").to(args.device)
            images = images.to(args.device)
            bs = images.shape[0]
            logits = model(images, text.input_ids, text.attention_mask)
            # print(logits.shape)
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            # print(indexs)
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return compute_ranks(eval_dataset,results)



def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result

def evaluate_fine(i2t_ranks, t2i_ranks):
    # i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def evaluate_coarse(eval_dataset, test_results):
    i2t_ranks, t2i_ranks, caption_index, image_index = compute_ranks_coarse(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result, caption_index, image_index


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
    parser.add_argument("--eval_img_keys_file", default='', type=str, 
                        help="image key tsv to select a subset of images for evaluation. "
                        "This is useful in 5-folds evaluation. The topn index file is not " 
                        "needed in this case.")
    parser.add_argument("--eval_caption_index_file", default='', type=str, 
                        help="index of a list of (img_key, cap_idx) for each image."
                        "this is used to perform re-rank using hard negative samples."
                        "useful for validation set to monitor the performance during training.")
    parser.add_argument("--cross_image_eval", action='store_true', 
                        help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument('--num_images_per_cap_val', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial lr.")
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
    parser.add_argument('--save_steps', type=int, default=-1, 
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
    parser.add_argument('--save_metric', type=str, default='r1', help='the validation metric for saving')
    parser.add_argument('--train_language', type=str, default=None, help='the language to train')
    parser.add_argument('--test_language', type=str, default=None, help='the language to test')
    parser.add_argument('--image_dir_format', type=str, default=None, help='the image dir format')
    parser.add_argument('--evaluate_before_training', action='store_true', help='whether to evaluate before training')
    # parser.add_argument('--data_file_train', type=str, default=None, help='the caption data file of training')
    # parser.add_argument('--data_file_val', type=str, default=None, help='the caption data file of validation')
    args = parser.parse_args()

    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        
    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, AutoTokenizer
    albef_config = yaml.load(open(args.albef_config, 'r'), Loader=yaml.Loader)
    bert_config = BertConfig.from_json_file(albef_config['bert_config'])

    # dataset loading
    if args.image_dir_format is None:
        args.image_dir = albef_config['image_root']
    else:
        args.image_dir = albef_config['image_root'][args.image_dir_format]
    
    if args.train_language is None or args.train_language=='all':
        args.data_file_train = albef_config['train_file']
    else:
        args.data_file_train = albef_config['train_file'][args.train_language]

    if args.test_language is None or args.test_language=='all':
        if args.eval_split == 'val':
            args.data_file_val = albef_config['val_file']
        elif args.eval_split == 'test':
            args.data_file_val = albef_config['test_file']
        else:
            raise NotImplementedError
    else:
        if args.eval_split == 'val':
            args.data_file_val = albef_config['val_file'][args.test_language]
        elif args.eval_split == 'test':
            args.data_file_val = albef_config['test_file'][args.test_language]
        else:
            raise NotImplementedError

    model_class = ALBEF
    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path)
        model = ALBEF(config=albef_config, text_encoder=albef_config['text_encoder'], tokenizer=tokenizer)
        args.dtype = torch.float32
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        try:
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
        except:
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = ALBEF(config=albef_config, text_encoder=albef_config['text_encoder'], tokenizer=tokenizer)
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
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
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
        if isinstance(args.data_file_train, dict):
            train_dataset = ConcatDataset([RetrievalDataset(caption_file=args.data_file_train[k], img_transform=train_transform, tokenizer=tokenizer, args=args, is_train=True) for k in args.data_file_train])
        else:
            train_dataset = RetrievalDataset(caption_file=args.data_file_train, img_transform=train_transform, tokenizer=tokenizer, args=args, is_train=True)
        if args.evaluate_during_training:
            if isinstance(args.data_file_val, dict):
                val_dataset = {k: RetrievalDataset(caption_file=v, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False) for k,v in args.data_file_val.items()}
            else:
                val_dataset = RetrievalDataset(caption_file=args.data_file_val, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            # args = restore_training_settings(args)
            test_dataset = RetrievalDataset(caption_file=args.data_file_val, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False)
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
                test_dataset = {k:RetrievalDataset(caption_file=v, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False) for k,v in args.data_file_val.items()}
            else:
                test_dataset = RetrievalDataset(caption_file=args.data_file_val, img_transform=val_transform, tokenizer=tokenizer, args=args, is_train=False)
            checkpoint = args.eval_model_dir
            assert op.isdir(checkpoint)
            logger.info("Evaluate the following checkpoint: %s", checkpoint)

            if args.half_evaluation:
                model = model.half()
                args.dtype = torch.float16
            else:
                args.dtype = torch.float32
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

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
                final_res = {}
                for k,v in test_dataset.items():
                    logger.info('evaluate on {}'.format(k))
                    coarse_sim = test_coarse(args, model, v, tokenizer=tokenizer)
                    eval_result, caption_index, image_index = evaluate_coarse(v, coarse_sim)
                    # caption index and image index
                    eval_i2t_result, _ = test_fine_i2t(args, model, v, caption_index=caption_index, tokenizer=tokenizer)
                    eval_t2i_result = test_fine_t2i(args, model, v, image_index=image_index, tokenizer=tokenizer)
                    print('fine inference:')
                    # print(eval_i2t_result, eval_t2i_result)
                    if args.do_eval:
                        eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)
                        final_res[k] = eval_result
                result_file = op.splitext(pred_file)[0] + '.eval.json'
                with open(result_file, 'w') as f:
                    json.dump(final_res, f)
                logger.info("Evaluation results saved to {}.".format(result_file))
            else:
                coarse_sim = test_coarse(args, model, test_dataset, tokenizer=tokenizer)
                eval_result, caption_index, image_index = evaluate_coarse(test_dataset, coarse_sim)
                # caption index and image index
                eval_i2t_result, _ = test_fine_i2t(args, model, test_dataset, caption_index=caption_index, tokenizer=tokenizer)
                eval_t2i_result = test_fine_t2i(args, model, test_dataset, image_index=image_index, tokenizer=tokenizer)
                print('fine inference:')
                # print(eval_i2t_result, eval_t2i_result)
                if args.do_eval:
                    eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)
                    result_file = op.splitext(pred_file)[0] + '.eval.json'
                    with open(result_file, 'w') as f:
                        json.dump(eval_result, f)
                    logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
