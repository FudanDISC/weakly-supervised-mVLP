from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import sys
import time
import math
import shutil
import deepspeed
import json

from torch.distributed.distributed_c10d import barrier
from torch.optim import lr_scheduler
from torch.utils.checkpoint import checkpoint
import yaml

sys.path.insert(0, '.')

import numpy as np
import torch
# from memory_profiler import profile

# from oscar.modeling.modeling_bert import BertImgForPreTraining, VLBertImgForPreTraining, BertImgForPreTraining3
from albef.modeling.model_pretrain import ALBEF, ALBEF_Stage1, ALBEF_fast
from albef.modeling.vit import interpolate_pos_embed
from albef.optim import create_optimizer
from albef.scheduler import create_scheduler
from albef import utils as albef_utils
from transformers_past.pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer)

from cvlm.datasets.img_ml_datasets.build import make_data_loader, make_data_loader_ds
from cvlm.datasets.parallel_txt_datasets.build import make_data_loader_ds as make_txt_data_loader
from cvlm.datasets.parallel_txt_datasets.build import make_mono_txt_loader_ds

from transformers_past.pytorch_transformers import AdamW, WarmupLinearSchedule
from utils.misc import mkdir, get_rank
from utils.metric_logger import TensorboardLogger
from utils.logger import setup_logger
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoTokenizer

# logger = logging.getLogger(__name__)
# logger = setup_logger("vlpretrain", )

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())


""" ****** Pretraining ****** """

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. "
                             "Should contain the .yaml files for the task.")
    parser.add_argument("--dataset_file", default=None, type=str, required=True,
                        help="The training dataset yaml file.")
    parser.add_argument("--extra_dataset_file", default=None, type=str, required=False,
                        help="The extra training dataset yaml file.")
    parser.add_argument('--albef_config', default=None, type=str, help='the albef config file path')
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    # image chunks
    parser.add_argument("--chunk_start_id", default=-1, type=int,
                        help="Image Chunk Start ID")
    parser.add_argument("--chunk_end_id", default=-1, type=int,
                        help="Image Chunk End ID")

    ## Image parameters
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str,
                        help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--use_layernorm", action='store_true',
                        help="use_layernorm")

    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out for BERT.")

    parser.add_argument("--use_b", type=int, default=1, help="use_b")
    parser.add_argument("--textb_sample_mode", type=int, default=0,
                        help="0: sample from both texta&textb, "
                             "1: sample from textb, "
                             "2: sample from QA answers")
    parser.add_argument("--extra_textb_sample_mode", type=int, default=1)
    parser.add_argument("--texta_false_prob", type=float, default=0.0,
                        help="the probality that we sample wrong texta, should in [0.0, 0.5]")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=35, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_seq_length_txt", default=35, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization for text pair dataset. \n"
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_iters", default=2000000, type=int,
                        help="Maximal number of training iterations.")
    parser.add_argument("--train_batch_size", default=1024, type=int,
                        help="Batch size for training.")
    parser.add_argument("--num_workers", default=6, type=int,
                        help="Number of workers for dataset.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--optim", default='adamw', type=str,
                        help="The optimizer used for Bert, [adamw, lamb], default: adamw")
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory", action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    parser.add_argument("--from_scratch", action='store_true',
                        help="train from scratch")
    parser.add_argument("--use_img_layernorm", type=int, default=0,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    # distributed
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument("--mask_loss_for_unmatched", type=int, default=1,
                        help="masked language model loss for unmatched triplets")
    parser.add_argument("--extra_loss_weight", type=float, default=0.0,
                        help="the loss weight for the extra train data batch (should be in [0,1])")
    parser.add_argument(
        "--use_gtlabels",
        type=int, default=1,
        help="use groundtruth labels for text b or not"
    )
    # logging
    parser.add_argument('--ckpt_period', type=int, default=10000,
                        help="Period for saving checkpoint")
    parser.add_argument('--log_period', type=int, default=100,
                        help="Period for saving logging info")
    parser.add_argument('--visual_learning', action='store_true',
                        help='whether to add visual learning')
    parser.add_argument('--tag2id', type=str, default=None,
                        help='object tag to id mapping')
    parser.add_argument('--text_corpus', type=str, default=None,
                        help='text-only corpus tsv file')
    parser.add_argument('--change_theme', action='store_true')
    parser.add_argument('--max_visual_themes', type=int, default=5, help='maximal number of visual theme concepts')
    parser.add_argument('--max_phrases', type=int, default=5, help='maximal number of phrase concepts')
    parser.add_argument('--mlm_debug', action='store_true', help='whether to output mlm result')
    parser.add_argument('--only_cap', type=int, default=0)
    parser.add_argument('--max_tag_length', type=int, default=20)
    parser.add_argument('--wra_mod', type=str, default='sample')
    parser.add_argument('--wra_layer', type=int, default=None, help='which join layer output is used for phrase grounding')
    parser.add_argument('--only_qa', action='store_true')
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--no_fk_test', action='store_true')
    parser.add_argument('--use_hungarian', action='store_true')
    parser.add_argument('--phrase_type_id', type=int, default=0, help='the segment ids of phrases')
    parser.add_argument('--phrase_mask', action='store_true', help='whther to use masked phrase modeling')
    parser.add_argument('--uni_lr', type=float, default=None, help='learning rate for uni-modal encoder')
    parser.add_argument('--num_readers', type=int, default=1, help='number of readers in arnold dataset')
    parser.add_argument('--display_time', action='store_true', help='display the detailed data loading time')
    parser.add_argument('--resume', action='store_true', help='resume from the checkpoint')
    parser.add_argument('--wk_sample', type=int, default=-1, help='the sub samples for wukong dataset')
    parser.add_argument('--txt_dataset_file', type=str, default=None, help='the text pair dataset yaml file')
    parser.add_argument('--train_batch_size_txt', type=int, default=1024, help='the training batch size for text pairs')
    parser.add_argument('--img_txt_mod', type=str, default='img-txt-contras', help='the pre-training mode for image text pairs')
    parser.add_argument('--txt_txt_mod', type=str, default='txt-txt-contras', help='the pre=training mode for text pair datasets')
    parser.add_argument('--txt_loss_weight', type=float, default=1.0)
    parser.add_argument('--data_debug', action='store_true', help='if only use part of the data to debug')
    parser.add_argument('--txt_dataformat', type=str, default='raw', help='the text dataset format')
    parser.add_argument('--train_batch_size_mono_txt', type=int, default=None)
    parser.add_argument('--mono_txt_dataset_file', type=str, default=None)
    parser.add_argument('--mono_txt_loss_weight', type=float, default=1.0)
    parser.add_argument('--mono_txt_max_length', type=int, default=64)
    parser.add_argument('--avoid_mlm_head_tie', action='store_true', help='whether to avoid mlm head tie')
    parser.add_argument('--mono_debug', action='store_true')
    parser.add_argument('--ablate_tlm', action='store_true', help='whether to ablate the translation MLM')

    # parser.add_argument('--deepspeed', action='store_true', help='whether to use deepspeed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)
    print("current rank:", args.local_rank)
    logger = setup_logger("vlpretrain", args.output_dir, args.local_rank)

    if args.text_corpus is not None:
        args.text_corpus = os.path.join(args.data_dir, args.text_corpus)

    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, 'r') as of:
            ds_config = json.load(of)
        if 'fp16' in ds_config:
            if ds_config['fp16']['enabled']:
                args.dtype = torch.float16
            else:
                args.dtype = torch.float32

    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    args.num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        if args.deepspeed:
            deepspeed.init_distributed()
        else:
            torch.distributed.init_process_group(
                backend='nccl', init_method="env://"
            )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.ERROR)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1)
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError(
            "Training is currently the only implemented execution option. Please set `do_train`.")

    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)

    last_checkpoint_dir = None
    arguments = {"iteration": 0}
    if os.path.exists(args.output_dir):
        save_file = os.path.join(args.output_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        if last_saved:
            folder_name = os.path.splitext(last_saved.split('/')[0])[0] # in the form of checkpoint-00001 or checkpoint-00001/pytorch_model.bin
            last_checkpoint_dir = os.path.join(args.output_dir, folder_name)
            arguments["iteration"] = int(folder_name.split('-')[-1])
            ALBEF_WEIGHT_NAME = 'ckpt.pth'
            assert os.path.isfile(os.path.join(last_checkpoint_dir, ALBEF_WEIGHT_NAME)), "Last_checkpoint detected, but file not found!"

    # model first
    if get_rank() != 0:
        torch.distributed.barrier()
    if last_checkpoint_dir is not None:  # recovery
        args.model_name_or_path = last_checkpoint_dir
        args.resume = True
        logger.info(" -> Recovering model from {}".format(last_checkpoint_dir))

    config = yaml.load(open(args.albef_config, 'r'), Loader=yaml.Loader)

    if 'conditional_mlm_probability' in config:
        logger.info("using a different conditional masked probability as {}".format(config['conditional_mlm_probability']))

    # # pre-pare dataset first
    # train_dataloaders = make_data_loader(
    #     args, is_distributed=args.distributed, arguments=arguments
    # )

    # Prepare model
    # model = BertForPreTraining.from_pretrained(args.bert_model)

    # prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)

    # train_dataloaders = make_data_loader_ds(
    #     args, is_distributed=args.distributed, arguments=arguments
    # )

    # if isinstance(train_dataloaders, list):
    #     train_dataloader = train_dataloaders[0]
    # else:
    #     train_dataloader = train_dataloaders
    # train_dataloader_extra = [None] * len(train_dataloader)
    # if isinstance(train_dataloaders, list) and len(train_dataloaders) > 1:
    #     logger.info("Having two train dataloaders!")
    #     train_dataloader_extra = train_dataloaders[1]
    # tokenizer = train_dataloader.dataset.tokenizer

    model = ALBEF_Stage1(config=config, text_encoder=config['text_encoder'], tokenizer=tokenizer, init_deit=True)
    logger.info("Training/evaluation parameters {}".format(config['avoid_nan']))
    bert_config = BertConfig.from_json_file(config['bert_config'])

    if args.model_name_or_path is not None:
        if args.model_name_or_path.endswith('.pth') or args.model_name_or_path.endswith('.pt'):
            ckpt_file = args.model_name_or_path
        else:
            ckpt_file = os.path.join(args.model_name_or_path, 'ckpt.pth')
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        logger.info(" -> Recovering model checkpoint from {}".format(ckpt_file))
        target_keys = set(model.state_dict().keys())
        # tmp_keys = ['text_encoder.bert.embeddings.word_embeddings.weight', 'text_encoder.cls.predictions.decoder.bias', 'text_encoder.cls.predictions.decoder.weight', 'text_encoder.cls.predictions.bias']
        tmp_keys = []
        try:
            state_dict = {k:v for k,v in checkpoint['model'].items() if k in target_keys and k not in tmp_keys}
        except:
            state_dict = {k:v for k,v in checkpoint.items() if k in target_keys and k not in tmp_keys}

        if model.visual_encoder.pos_embed.shape[1] != state_dict['visual_encoder.pos_embed'].shape[1]:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)     
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if model.text_encoder.bert.embeddings.word_embeddings.weight.shape[0] != state_dict['text_encoder.bert.embeddings.word_embeddings.weight'].shape[0]:
            # re-initialize the embedding weight and prediction head for embeddings of a new size
            logger.info('found the word embedding size is not aligned, re-initializing! {} vs {}'.format(model.text_encoder.bert.embeddings.word_embeddings.weight.shape[0], state_dict['text_encoder.bert.embeddings.word_embeddings.weight'].shape[0]))
            target_size = model.text_encoder.bert.embeddings.word_embeddings.weight.shape
            new_embs = torch.randn(target_size)*bert_config.initializer_range
            new_bias = torch.zeros(target_size[0])
            state_dict['text_encoder.bert.embeddings.word_embeddings.weight'] = new_embs
            state_dict['text_encoder.cls.predictions.decoder.bias'] = new_bias
            state_dict['text_encoder.cls.predictions.decoder.weight'] = new_embs
            state_dict['text_encoder.cls.predictions.bias'] = new_bias
        if not args.resume and not args.avoid_mlm_head_tie:
            # important here
            logger.info('tie the weight of mlm head and embedding')
            target_size = model.text_encoder.bert.embeddings.word_embeddings.weight.shape
            new_bias = torch.zeros(target_size[0])
            state_dict['text_encoder.cls.predictions.decoder.bias'] = new_bias
            state_dict['text_encoder.cls.predictions.decoder.weight'] = state_dict['text_encoder.bert.embeddings.word_embeddings.weight']
            state_dict['text_encoder.cls.predictions.bias'] = new_bias
        if hasattr(model, 'half_mlm_head'):
            if model.text_modeling:
                logger.info('found half mlm head')
                tmp_pair = []
                for k,v in state_dict.items():
                    if k.startswith('text_encoder.cls'):
                        tmp_pair.append([k, v])
                for p in tmp_pair:
                    k,v = p        
                    state_dict['half_mlm_head'+k[16:]] = v
        msg = model.load_state_dict(state_dict, strict=False)
        # print(model.txt_teacher_model.transformer.position_embeddings.weight)
        logger.info(msg)
        print('loaded checkpoint from {}'.format(args.model_name_or_path))

    for module_name in config['freeze_module']:
        logger.info('freezing the {} module'.format(module_name))
        model.freeze(module_name)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        'Total Parameters: {}'.format(total_params))

    if get_rank() == 0 and args.local_rank != -1:
        torch.distributed.barrier()

    if not args.deepspeed:
        model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    tb_log_dir = os.path.join(args.output_dir, 'train_logs')
    meters = TensorboardLogger(
        log_dir=tb_log_dir,
        delimiter="  ",
    )
    # meters = TensorboardLogger(
    #     log_dir='hdfs://haruna/home/byte_arnold_lq/data/ecom/kg_mm_cls/tasks/1542909/trials/3944048/output',
    #     delimiter='  ',
    # )

    arg_opt = albef_utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = albef_utils.AttrDict(config['scheduler'])
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup_steps,
                                     t_total=args.max_iters)
    # scheduler, _ = create_scheduler(arg_sche, optimizer)

    if arguments['iteration'] > 0 and args.resume:  # recovery
        logger.info(
            "Load BERT optimizer from {}".format(last_checkpoint_dir))
        # optimizer_to_load = torch.load(
        #     os.path.join(last_checkpoint_dir, 'optimizer.pth'),
        #     map_location=torch.device("cpu"))
        # optimizer.load_state_dict(optimizer_to_load.pop("optimizer"))
        # scheduler.load_state_dict(optimizer_to_load.pop("scheduler"))
        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # except:
        #     optimizer.load_state_dict(checkpoint['optimizer']['base_optimizer_state'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.deepspeed:
        model_engine, optimizer, _, scheduler = deepspeed.initialize(args=args, model=model, optimizer=optimizer, lr_scheduler=scheduler)
        # if last_checkpoint_dir is not None:
        #     model_engine.load_checkpoint(last_checkpoint_dir)
        #     optimizer = model_engine.optimizer
        #     scheduler = model_engine.lr_scheduler
        model = model_engine.module
        print(optimizer)
        print(scheduler)
    elif args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # image-text pair
    train_dataloaders = make_data_loader_ds(
        args, tokenizer=tokenizer, is_distributed=args.distributed, arguments=arguments
    )

    if isinstance(train_dataloaders, list):
        train_dataloader = train_dataloaders[0]
    else:
        train_dataloader = train_dataloaders
    train_dataloader_extra = [None] * len(train_dataloader)
    if isinstance(train_dataloaders, list) and len(train_dataloaders) > 1:
        logger.info("Having two train dataloaders!")
        train_dataloader_extra = train_dataloaders[1]

    # text pair dataset
    if args.ablate_tlm:
        txt_dataloader = [None] * len(train_dataloader)
    else:
        txt_dataloaders = make_txt_data_loader(
            args, tokenizer, is_distributed=args.distributed, arguments=arguments
        )
        txt_dataloader = txt_dataloaders[0]

    # mono text data loader
    if args.mono_debug:
        mono_txt_loader = [None] * len(train_dataloader)
    else:
        mono_txt_loader = make_mono_txt_loader_ds(
            args, tokenizer, is_distributed=args.distributed, arguments=arguments
        )

    # torch.backends.cudnn.benchmark = True

    max_iter = len(train_dataloader)
    start_iter = arguments["iteration"]
    # logging.disable(logging.WARNING)
    logger.info("***** Running training *****")
    logger.info(" Num examples = {}".format(len(train_dataloader.dataset)))
    logger.info("  Instantaneous batch size = %d",
                args.train_batch_size // args.gradient_accumulation_steps)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d",
                max_iter // args.gradient_accumulation_steps)

    log_json = {}

    if args.deepspeed:
        model_engine.train()
        model_engine.zero_grad()
    else:
        model.train()
        model.zero_grad()

    clock_started = False
    # Every args.ckpt_period, report train_score and save model
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    if args.mlm_debug:
        tmp_mlm = []

    if args.txt_txt_mod == 'txt-txt-tlm' or args.txt_txt_mod == 'txt-txt-contras' or args.txt_txt_mod == 'para_txt_full' or args.txt_txt_mod == 'para_txt_abl-xcl':
        extra_txt_loss_name = 'tlm'
    elif args.txt_txt_mod == 'txt-txt-wla':
        extra_txt_loss_name = 'wla'
    else:
        extra_txt_loss_name = 'unkown'

    if args.txt_txt_mod == 'txt-txt-wla':
        assert 'wla_modeling' in config
        wla_args = {'token_sim_mode':config['wla_modeling']['mod'], 'token_sim_k':config['wla_modeling']['k']}
    for step, (batch, batch_extra, batch_txt, batch_mono) in enumerate(zip(train_dataloader, train_dataloader_extra, txt_dataloader, mono_txt_loader), start_iter):
        if not clock_started:
            start_training_time = time.time()
            end = time.time()
            clock_started = True

        def data_process(mini_batch):
            # images, targets, qa_inds = \
            #     mini_batch[0], mini_batch[1], mini_batch[2]
            # targets_transposed = list(zip(*targets))
            # print(mini_batch)
            images = mini_batch[0]
            texts = mini_batch[1]

            if args.deepspeed:
                c_device = model_engine.device
            else:
                c_device = args.device
            text_input = tokenizer(texts, padding='longest', truncation=True, max_length=args.max_seq_length, return_tensors="pt").to(c_device)
            # images = torch.stack(images).to(dtype=args.dtype, device=model_engine.device, non_blocking=True)
            # my_time_0 = time.time()
            # tmp_shape = images[0].shape
            # images = torch.cat(mini_batch[0]).reshape(-1, 50, 2054)
            # my_time_1 = time.time()
            if images is not None:
                images = images.to(c_device, non_blocking=True)
                # my_time_2 = time.time()
                images = images.to(dtype=args.dtype)
            # my_time_3 = time.time()
            # print('stack time {}, cuda time {}, half time {}'.format(my_time_1-my_time_0, my_time_2-my_time_1, my_time_3-my_time_2))
            # if isinstance(images, torch.Tensor):
            #     images = images.to(args.dtype)
            #     images = images.to(c_device, non_blocking=True)
            # else:
            #     images = [i.to(dtype=args.dtype, device=c_device, non_blocking=True) for i in images]

            return images, text_input

        def txt_data_process(mini_batch):
            if args.txt_dataformat == 'raw':
                src_txts = mini_batch[0]
                tar_txts = mini_batch[1]
                src_lang_ids = mini_batch[2].to(c_device)
                tar_lang_ids = mini_batch[3].to(c_device)
            elif args.txt_dataformat == 'transformers':
                src_txts = mini_batch['source_text']
                tar_txts = mini_batch['target_text']
                src_lang_ids = None
                tar_lang_ids = None

            # if get_rank() == 0:
            #     print(src_txts)
            #     print(tar_txts)

            if args.deepspeed:
                c_device = model_engine.device
            else:
                c_device = args.device
            src_text_input = tokenizer(src_txts, padding='longest', truncation=True, max_length=args.max_seq_length_txt, return_tensors="pt").to(c_device)
            tar_text_input = tokenizer(tar_txts, padding='longest', truncation=True, max_length=args.max_seq_length_txt, return_tensors="pt").to(c_device)

            return src_text_input, tar_text_input, src_lang_ids, tar_lang_ids

        def mono_txt_data_process(mini_batch):
            txts = mini_batch['text']

            if args.deepspeed:
                c_device = model_engine.device
            else:
                c_device = args.device
            text_input = tokenizer(txts, padding='longest', truncation=True, max_length=args.mono_txt_max_length, return_tensors="pt").to(c_device)

            return text_input

        images1, text_input1 = data_process(batch)
        if batch_txt is None:
            src_text = None
        else:
            src_text, tar_text, src_lang_ids, tar_lang_ids = txt_data_process(batch_txt)
        if batch_mono is None:
            mono_txt = None
        else:
            mono_txt = mono_txt_data_process(batch_mono)
        # return None
        # print(images1, text_input1)
        # print(src_text, tar_text)
        # return None

        data_time = time.time() - end

        def forward_backward(image_features, text_input, loss_weight=1.0):
            if args.deepspeed:
                # print(image_features.shape, input_ids.shape, lm_label_ids.shape)
                outputs = model_engine(image=image_features, text=text_input, mode=args.img_txt_mod)
            else:
                outputs = model(image=image_features, text=text_input, mode=args.img_txt_mod)

            # torch.save(outputs, '/opt/tiger/debug/test_output.pt')
            if len(outputs) == 3:
                loss_mlm, loss_ita, loss_itm = outputs
                loss = loss_weight * (loss_mlm + loss_ita + loss_itm)
                loss_mlm = loss_weight * loss_mlm.item()
                loss_ita = loss_weight * loss_ita.item()
                loss_itm = loss_weight * loss_itm.item()
            else:
                loss_ita = outputs[0]
                loss = loss_ita * loss_weight
                loss_mlm = 0
                loss_ita = loss_weight * loss_ita.item()
                loss_itm = 0

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.deepspeed:
                model_engine.backward(loss)
            else:
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            return loss.item(), image_features.size(0), loss_mlm, loss_ita, loss_itm


        def forward_backward_txt(text_input, trans_text_input, text_lang=None, trans_lang=None, loss_weight=1.0):
            if args.deepspeed:
                # print(image_features.shape, input_ids.shape, lm_label_ids.shape)
                if args.txt_txt_mod == 'txt-txt-wla':
                    outputs = model_engine(text=text_input, text_trans=trans_text_input, text_langs=text_lang, text_trans_langs=trans_lang, mode=args.txt_txt_mod, **wla_args)
                else:
                    outputs = model_engine(text=text_input, text_trans=trans_text_input, text_langs=text_lang, text_trans_langs=trans_lang, mode=args.txt_txt_mod)

            else:
                if args.txt_txt_mod == 'txt-txt-wla':
                    outputs = model(text=text_input, text_trans=trans_text_input, text_langs=text_lang, text_trans_langs=trans_lang, mode=args.txt_txt_mod, **wla_args)
                else:
                    outputs = model(text=text_input, text_trans=trans_text_input, text_langs=text_lang, text_trans_langs=trans_lang, mode=args.txt_txt_mod)
                # outputs = model(text=text_input, text_trans=trans_text_input, text_langs=text_lang, text_trans_langs=trans_lang, mode=args.txt_txt_mod)

            # print(outputs)
            if len(outputs) == 1:
                # only contrastive loss
                loss_ita_txt = outputs[0]
                loss = loss_weight * loss_ita_txt
                loss_ita_txt = loss_weight * loss_ita_txt.item()
                loss_tlm_txt = 0.0
            else:
                if args.txt_txt_mod == 'txt-txt-tlm':
                    loss_ita_txt, loss_tlm_txt = outputs
                    loss = loss_weight * (loss_ita_txt + loss_tlm_txt)
                    loss_ita_txt = loss_weight * loss_ita_txt.item()
                    loss_tlm_txt = loss_weight * loss_tlm_txt.item()
                elif args.txt_txt_mod == 'txt-txt-wla':
                    loss_ita_txt, loss_tlm_txt = outputs # tlm here but actually the weak-aligment loss
                    loss = loss_weight * (loss_ita_txt + loss_tlm_txt)
                    loss_ita_txt = loss_weight * loss_ita_txt.item()
                    loss_tlm_txt = loss_weight * loss_tlm_txt.item()
                elif args.txt_txt_mod == 'para_txt_full':
                    loss_ita_txt, loss_tlm_s2t, loss_tlm_t2s = outputs
                    loss = loss_weight * (loss_ita_txt + loss_tlm_s2t + loss_tlm_t2s)
                    loss_ita_txt = loss_weight * loss_ita_txt.item()
                    loss_tlm_txt = loss_weight * (loss_tlm_s2t + loss_tlm_t2s).item()
                elif args.txt_txt_mod == 'para_txt_abl-xcl':
                    loss_tlm_s2t, loss_tlm_t2s = outputs
                    loss = loss_weight * (loss_tlm_s2t + loss_tlm_t2s)
                    loss_ita_txt = 0.0
                    # loss_ita_txt = loss_weight * loss_ita_txt.item()
                    loss_tlm_txt = loss_weight * (loss_tlm_s2t + loss_tlm_t2s).item()
                else:
                    raise NotImplementedError

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.deepspeed:
                model_engine.backward(loss)
            else:
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            return loss.item(), text_input.input_ids.size(0), loss_ita_txt, loss_tlm_txt

        def forward_backward_mono_txt(text_input, loss_weight=1.0):
            if args.deepspeed:
                outputs = model_engine(text=text_input, mode='mono_txt')
            else:
                outputs = model(text=text_input, mode='mono_txt')
                # outputs = model(text=text_input, text_trans=trans_text_input, text_langs=text_lang, text_trans_langs=trans_lang, mode=args.txt_txt_mod)

            mono_mlm_loss = outputs[0]
            loss = loss_weight * mono_mlm_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.deepspeed:
                model_engine.backward(loss)
            else:
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            return loss.item(), text_input.input_ids.size(0)


        start1 = time.time()

        # loss1, nb_tr_example1, wra_loss1, vis_mlm_loss1, retrieval_loss1, mlm_loss1, relation_loss1, qa_rel_loss1, phrase_mask_loss1 = forward_backward(
        #         image_features=images1, input_ids_a=input_ids_a1, input_mask_a=input_mask_a1, qa_is_next=is_next1,
        #         segment_ids_a=segment_ids_a1, lm_label_ids_a=lm_label_ids_a1, input_ids_b=input_ids_b1,
        #         input_mask_b=input_mask_b1, segment_ids_b=segment_ids_b1, lm_label_ids_b=lm_label_ids_b1,
        #         loss_weight=1.0-args.extra_loss_weight, phrase_index=phrase_index, image_index=image_index, phrase_mask_label=phrase_mask_label1
        #     )
        loss1, nb_tr_example1, mlm_loss1, ita_loss1, itm_loss1 = forward_backward(image_features=images1, text_input=text_input1, loss_weight=1.0)
        # return None
        tr_loss += loss1
        nb_tr_examples += nb_tr_example1
        # print(nb_tr_example1)

        start2 = time.time()
        compute_time1 = start2 - start1
        loss2, nb_tr_example2 = 0.0, 0
        ita_loss2, mlm_loss2, itm_loss2 = 0.0, 0.0, 0.0

        # text pairs forward
        if src_text is None:
            loss_txt, loss_t2t, loss_tlm = 0.0, 0.0, 0.0
        else:
            loss_txt, nb_txt_example, loss_t2t, loss_tlm = forward_backward_txt(text_input=src_text, trans_text_input=tar_text, text_lang=src_lang_ids, trans_lang=tar_lang_ids, loss_weight=args.txt_loss_weight)

        start3 = time.time()
        compute_time2 = start3 - start2

        if mono_txt is None:
            loss_mono = 0.0
        else:
            loss_mono, nb_mono_example = forward_backward_mono_txt(mono_txt, loss_weight=args.mono_txt_loss_weight)
        compute_time3 = time.time() - start3

        

        nb_tr_steps += 1
        arguments["iteration"] = step + 1

        if args.deepspeed:
            model_engine.step()
        # print(step)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # do gradient clipping
            if not args.deepspeed:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # do the optimization steps
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            metrics_to_log = {
                'time_info': {'compute': batch_time/args.gradient_accumulation_steps, 'data': data_time,
                              'compute1': compute_time1,
                              'compute2': compute_time2, 'compute3': compute_time3},
                'batch_metrics': {'loss': loss1+loss2+loss_txt+loss_mono, 'mlm_loss':mlm_loss1+mlm_loss2, 'ita_loss':ita_loss1+ita_loss2, 
                        'itm_loss': itm_loss1+itm_loss2, 't2t_loss': loss_t2t, '{}_loss'.format(extra_txt_loss_name): loss_tlm, 'mono_mlm': loss_mono}
            }
            # print(metrics_to_log)
            params_to_log = {'params': {'bert_lr': optimizer.param_groups[0]["lr"]}}
            meters.update_metrics(metrics_to_log)
            meters.update_params(params_to_log)
            # return None

            if args.log_period > 0 and (step + 1) % args.log_period == 0:
                # if args.mlm_debug and args.local_rank==0:
                #     tmp_path = os.path.join(args.output_dir,
                #                           'mlm_res.pt')
                #     torch.save(tmp_mlm, tmp_path)
                # barrier()
                # return None
                avg_time = meters.meters['time_info']['compute'].global_avg
                eta_seconds = avg_time * (max_iter - step - 1)
                eta_string = str(
                    datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=step + 1,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    ) + "\n    " + meters.get_logs(step + 1)
                )

        if (step + 1) == max_iter or (step + 1) % args.ckpt_period == 0:  # Save a trained model
            log_json[step+1] = tr_loss
            train_metrics_total = torch.Tensor([tr_loss, nb_tr_examples, nb_tr_steps]).to(args.device)
            torch.distributed.all_reduce(train_metrics_total)
            # reset metrics
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if get_rank() == 0:
                # report metrics
                train_score_gathered = train_metrics_total[0] / \
                                       train_metrics_total[2]
                logger.info("PROGRESS: {}%".format(
                    round(100 * (step + 1) / max_iter, 4)))
                logger.info(
                    "EVALERR: {}%".format(train_score_gathered))
                meters.update_metrics(
                    {
                        'epoch_metrics': {'ex_cnt': train_metrics_total[1],
                                          'loss': train_score_gathered}
                    }
                )
                with open(os.path.join(args.output_dir, 'loss_logs.json'),
                          'w') as fp:
                    json.dump(log_json, fp)

                # save checkpoint
                output_dir = os.path.join(args.output_dir,
                                          'checkpoint-{:07d}'.format(
                                              step + 1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args.deepspeed:
                    model_to_save = model
                else:
                    model_to_save = model.module if hasattr(
                        model,
                        'module') else model  # Take care of distributed/parallel training

                save_num = 0
                while save_num < 10:
                    print('save the', save_num, 'times')
                    try:
                        torch.save(args, os.path.join(output_dir,
                                                      'training_args.bin'))
                        tokenizer.save_pretrained(output_dir)
                        object_to_save = {
                            'model': model_to_save.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'config': config,
                            'step': step + 1 
                        }
                        torch.save(object_to_save,
                                   os.path.join(output_dir,
                                                'ckpt.pth'))
                        save_file = os.path.join(args.output_dir, "last_checkpoint")
                        with open(save_file, "w") as f:
                            f.write('checkpoint-{:07d}/ckpt.pth'.format(step + 1))
                        break
                    except:
                        save_num += 1
                logger.info(
                    "Saving model checkpoint {0} to {1}".format(
                        step + 1, output_dir))

            output_dir = os.path.join(args.output_dir,
                                          'checkpoint-{:07d}'.format(
                                              step + 1))
            if args.deepspeed:
                model_engine.save_checkpoint(output_dir)

    if clock_started:
        total_training_time = time.time() - start_training_time
    else:
        total_training_time = 0.0
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )
    # close the tb logger
    meters.close()


if __name__ == "__main__":
    main()