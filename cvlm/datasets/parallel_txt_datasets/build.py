import os
import logging
import torch
from oscar.utils.misc import get_world_size
from torch.utils.data import dataset
from .oscar_tsv import OscarTSVDataset, ImgOnlyDataset, TextOnlyDataset, TextOnlyDataset2
from .oscar_tsv_txt import ParallelTxtDataset
from transformers_past.pytorch_transformers import BertTokenizer
from transformers import AutoTokenizer
from torchvision import transforms
from oscar.utils.randaugment import RandomAugment
from PIL import Image
import yaml
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments
from oscar.utils.sampler_utils import MyDistributedSampler


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    """
    def __call__(self, batch):
        return list(zip(*batch))


def build_dataset(args, tokenizer):
    """
    Arguments:
        args: configuration.
    """
    full_yaml_file = os.path.join(args.data_dir, args.txt_dataset_file)
    assert os.path.isfile(full_yaml_file)
    config = yaml.load(open(full_yaml_file, 'r'), Loader=yaml.Loader)

    cfg = dict(
        config=config,
        root=args.data_dir,
        args=args,
        seq_len=args.max_seq_length,
        on_memory=args.on_memory,
        tokenizer=tokenizer,
    )
    # make dataset from factory
    datasets = [ParallelTxtDataset(**cfg)]
    # datasets = [ImgOnlyDataset(**cfg)]
    # datasets = [TextOnlyDataset2(input_tsv=args.text_corpus, args=args, seq_len=args.max_seq_length, tokenizer=tokenizer)]
    if args.extra_dataset_file:
        full_yaml_file = os.path.join(args.data_dir, args.extra_dataset_file)
        assert os.path.isfile(full_yaml_file)
        cfg['yaml_file'] = full_yaml_file
        cfg['textb_sample_mode'] = args.extra_textb_sample_mode
        datasets.append(OscarTSVDataset(**cfg))

    return datasets

def build_full_dataset(args):
    """
    Arguments:
        args: configuration.
    """
    full_yaml_file = os.path.join(args.data_dir, args.dataset_file)
    assert os.path.isfile(full_yaml_file)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)

    cfg = dict(
        yaml_file=full_yaml_file,
        args=args,
        seq_len=args.max_seq_length,
        on_memory=args.on_memory,
        tokenizer=tokenizer,
    )
    # make dataset from factory
    datasets = [OscarTSVDataset(**cfg), ImgOnlyDataset(**cfg)]
    datasets.append(TextOnlyDataset2(input_tsv=args.text_corpus, args=args, seq_len=args.max_seq_length, tokenizer=tokenizer))
    if args.extra_dataset_file:
        full_yaml_file = os.path.join(args.data_dir, args.extra_dataset_file)
        assert os.path.isfile(full_yaml_file)
        cfg['yaml_file'] = full_yaml_file
        cfg['textb_sample_mode'] = args.extra_textb_sample_mode
        datasets.append(OscarTSVDataset(**cfg))

    return datasets


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        # return MyDistributedSampler(dataset, shuffle=shuffle)
        return torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(
        sampler, images_per_batch, num_iters=None,
        start_iter=0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(args, tokenizer, is_distributed=False, arguments=None):
    num_gpus = get_world_size()
    # figure out start iteration
    if arguments is None:
        start_iter = 0
    else:
        start_iter = arguments['iteration']
    # figure out the batchsize
    grad_accumulate_steps = 1
    if hasattr(args, 'gradient_accumulation_steps'):
        grad_accumulate_steps = args.gradient_accumulation_steps
    assert (
            args.train_batch_size % grad_accumulate_steps == 0
    ), "train_batch_size ({}) must be divisible by the number "
    "of Gradient accumulation ({}) used."\
        .format(args.train_batch_size, grad_accumulate_steps)
    images_per_batch = args.train_batch_size//grad_accumulate_steps
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    logger = logging.getLogger(__name__)
    logger.info("Train with {} images per GPU".format(images_per_gpu))
    shuffle = True
    num_iters = args.max_iters * grad_accumulate_steps

    # build dataset
    datasets = build_dataset(args, tokenizer=tokenizer)

    data_loaders = []
    for i, dataset in enumerate(datasets):
        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
           sampler, images_per_gpu, num_iters, start_iter
        )
        num_workers = args.num_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(),
            pin_memory=True,
        )
        data_loaders.append(data_loader)
    return data_loaders

def make_data_loader_ds(args, tokenizer, is_distributed=False, arguments=None):
    num_gpus = get_world_size()
    # figure out start iteration
    if arguments is None:
        start_iter = 0
    else:
        start_iter = arguments['iteration']
    # figure out the batchsize
    grad_accumulate_steps = 1
    if hasattr(args, 'gradient_accumulation_steps'):
        grad_accumulate_steps = args.gradient_accumulation_steps
    assert (
            args.train_batch_size_txt % grad_accumulate_steps == 0
    ), "train_batch_size ({}) must be divisible by the number "
    "of Gradient accumulation ({}) used."\
        .format(args.train_batch_size_txt, grad_accumulate_steps)
    texts_per_batch = args.train_batch_size_txt
    assert (
        texts_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(texts_per_batch, num_gpus)
    texts_per_gpu = texts_per_batch // num_gpus
    logger = logging.getLogger(__name__)
    logger.info("Train with {} text pairs per GPU".format(texts_per_gpu))
    shuffle = True
    num_iters = args.max_iters

    # build dataset
    if args.txt_dataformat == 'raw':
        datasets = build_dataset(args, tokenizer=tokenizer)
    elif args.txt_dataformat == 'transformers':
        datasets = build_transformer_dataset(args)
    else:
        raise NotImplementedError

    logger.info('using parallel datasets with {} pairs'.format(len(datasets[0])))

    data_loaders = []
    for i, dataset in enumerate(datasets):
        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
           sampler, texts_per_gpu, num_iters, start_iter
        )
        num_workers = args.num_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            pin_memory=True
        )
        data_loaders.append(data_loader)
    return data_loaders

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    dataset.img_reader = {}
    dataset.od_reader = {}
    for k, i_d in dataset.image_feature_path.items():
        dataset.img_reader[k] = KVReader(i_d, dataset.num_readers)
    for k, o_d in dataset.image_label_path.items():
        dataset.od_reader[k] = KVReader(o_d, dataset.num_readers)


def make_full_data_loader_ds(args, is_distributed=False, arguments=None):
    # make all 3 modalities dataset in a row
    num_gpus = get_world_size()
    # figure out start iteration
    if arguments is None:
        start_iter = 0
    else:
        start_iter = arguments['iteration']
    # figure out the batchsize
    grad_accumulate_steps = 1
    if hasattr(args, 'gradient_accumulation_steps'):
        grad_accumulate_steps = args.gradient_accumulation_steps
    assert (
            args.train_batch_size % grad_accumulate_steps == 0
    ), "train_batch_size ({}) must be divisible by the number "
    "of Gradient accumulation ({}) used."\
        .format(args.train_batch_size, grad_accumulate_steps)
    images_per_batch = args.train_batch_size
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    logger = logging.getLogger(__name__)
    logger.info("Train with {} images per GPU".format(images_per_gpu))
    shuffle = True
    num_iters = args.max_iters

    # build dataset
    datasets = build_full_dataset(args)

    data_loaders = []
    dl_batch_size = [images_per_gpu, images_per_gpu//5, images_per_gpu//5]
    for i, dataset in enumerate(datasets):
        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
           sampler, dl_batch_size[i], num_iters, start_iter
        )
        num_workers = args.num_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        data_loaders.append(data_loader)
    return data_loaders


def build_transformer_dataset(args):
    """
    Arguments:
        args: configuration.
    """
    full_yaml_file = os.path.join(args.data_dir, args.txt_dataset_file)
    assert os.path.isfile(full_yaml_file)
    config = yaml.load(open(full_yaml_file, 'r'), Loader=yaml.Loader)

    assert config['format'] == 'transformers', 'the config file is not a transformer config, please check!'

    if config['corpus_file']['method'] == 'load_dataset':
        dataset = load_dataset('json', data_files=config['corpus_file']['path'])
    elif config['corpus_file']['method'] == 'load_from_disk':
        dataset = load_from_disk(config['corpus_file']['path'])['train']
    else:
        print('{} not supported yet'.format(config['corpus_file']['method']))
        raise NotImplementedError

    # dataset = dataset.select(range(10000))
    if hasattr(args, 'valid_langs'):
        valid_langs = set(args.valid_langs)
        dataset = dataset.filter(lambda x: x['target'] in valid_langs)
    
    elif config['languages'] != 'all':
        valid_langs = set(config['languages'].split(','))
        dataset = dataset.filter(lambda x: x['target'] in valid_langs)

    # datasets = [ImgOnlyDataset(**cfg)]
    # datasets = [TextOnlyDataset2(input_tsv=args.text_corpus, args=args, seq_len=args.max_seq_length, tokenizer=tokenizer)]

    return [dataset]


def make_mono_txt_loader_ds(args, tokenizer, is_distributed=False, arguments=None):
    num_gpus = get_world_size()
    # figure out start iteration
    if arguments is None:
        start_iter = 0
    else:
        start_iter = arguments['iteration']
    # figure out the batchsize
    grad_accumulate_steps = 1
    if hasattr(args, 'gradient_accumulation_steps'):
        grad_accumulate_steps = args.gradient_accumulation_steps
    assert (
            args.train_batch_size_mono_txt % grad_accumulate_steps == 0
    ), "train_batch_size ({}) must be divisible by the number "
    "of Gradient accumulation ({}) used."\
        .format(args.train_batch_size_mono_txt, grad_accumulate_steps)
    texts_per_batch = args.train_batch_size_mono_txt
    assert (
        texts_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(texts_per_batch, num_gpus)
    texts_per_gpu = texts_per_batch // num_gpus
    logger = logging.getLogger(__name__)
    logger.info("Train with {} mono sentences per GPU".format(texts_per_gpu))
    shuffle = True
    logger.info("Mono text with shuffle={}".format(shuffle))
    num_iters = args.max_iters

    # build dataset
    dataset = build_transformer_dataset_mono(args)

    logger.info('using monolingual datasets with {} sentences'.format(len(dataset)))


    # if is_distributed:
    #     sampler = MyDistributedSampler(dataset=dataset, shuffle=shuffle)
    # elif shuffle:
    #     sampler = torch.utils.data.sampler.RandomSampler(dataset)
    # else:
    #     sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)

    batch_sampler = make_batch_data_sampler(
           sampler, texts_per_gpu, num_iters, start_iter
        )
    num_workers = args.num_workers
    data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            pin_memory=True
        )
    return data_loader


def build_transformer_dataset_mono(args):
    """
    Arguments:
        args: configuration.
    """
    full_yaml_file = os.path.join(args.data_dir, args.mono_txt_dataset_file)
    assert os.path.isfile(full_yaml_file)
    config = yaml.load(open(full_yaml_file, 'r'), Loader=yaml.Loader)

    assert config['format'] == 'transformers', 'the config file is not a transformer config, please check!'

    dataset = load_from_disk(config['corpus_file']['path'])['train'] # .shuffle()

    return dataset