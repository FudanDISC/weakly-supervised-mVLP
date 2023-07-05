import collections
import os
import time
import json
import logging
import random
import glob
import base64
# from datasets.fingerprint import get_datasets_with_cache_file_in_temp_dir
from torch.nn.functional import GRID_SAMPLE_PADDING_MODES
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file
from collections import Counter
from transformers import XLMTokenizer
from PIL import Image
from collections import Counter
import pyarrow as pa
import lmdb
from io import BytesIO
from pathlib import Path

phrase_vocab_size = None
theme_vocab_size = None
mask_prob = 0.15
phrase_type_id = 0
concept_id2key = {}
all_concept_ids = []
only_multi_phrase = False
cls_token = '[CLS]'

class OscarJsonDatasetImgOD(Dataset):
    # Json based image-text dataset
    def __init__(self, root, config, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8",  on_memory=True, ds_names=None, transform=None,
                 **kwargs):
        self.cfg = config
        self.root = root
        self.tokenizer = tokenizer
        global cls_token
        if isinstance(self.tokenizer, XLMTokenizer):
            cls_token = self.tokenizer.bos_token
        else:
            cls_token = self.tokenizer.cls_token
        self.seq_len = seq_len
        self.transform = transform
        self.on_memory = on_memory
        global mask_prob
        mask_prob = args.mask_prob
        self.languages = args.languages
        self.sep_token = {'zh': ' '}
        logging.info('mask with {} probabilitiy'.format(mask_prob))

        if ds_names is None:
            self.only_image = False
            self.datasets_names = self.cfg['corpus'].split('_')
        else:
            self.only_image = True
            self.datasets_names = ds_names.split('_')

        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_path = self.cfg['image_path']
        for k,v in self.image_path.items():
            if v.startswith('/'):
                # absolute path
                self.image_path[k] = v
            else:
                # relative path
                self.image_path[k] = os.path.join(self.root, v)

        self.encoding = encoding
        self.dataset_with_splits = ['coco']
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # load samples into memory
        fk_count = 0
        # for checking the image loading information!
        self.img_info_counter = Counter()
        if on_memory:
            self.all_docs = []
            self.tuple_mask_map = []
            self.all_qa_docs = []
            self.all_qa_ans = []
            self.imgid2labels = {}
            self.label_map = {}
            self.corpus_lines = 0
            self.wk_count = 0
            max_tokens = 0
            self.debug_flag = False
            # tmp_tag2id = Counter()
            for dataset_name, corpus_info in self.cfg['corpus_file'].items():
                if dataset_name not in self.datasets_names:
                    continue
                corpus_file = corpus_info['path']
                corpus_format = corpus_info['format']
                if 'label_map' in corpus_info:
                    tmp_map = {}
                    for lang in self.languages:
                        labelmap_path = corpus_info['label_map'][lang]
                        labelmap_path = labelmap_path if labelmap_path.startswith('/') else os.path.join(self.root, labelmap_path)
                        tmp_map[lang] = json.load(open(labelmap_path, 'r'))
                    self.label_map[dataset_name] = tmp_map
                if corpus_format == 'json':
                    corpus_iter = self.load_json_corpus(corpus_info)
                elif corpus_format == 'data':
                    corpus_iter = self.load_data_corpus(corpus_info)
                else:
                    raise NotImplementedError
                for line in tqdm(corpus_iter, desc='loading the {} dataset'.format(dataset_name)):
                    img_id, objects = line
                    img_info = '{}|{}'.format(dataset_name, img_id)
                    if not self.check_img_exists(img_info):
                        continue
                    self.img_info_counter.update([dataset_name])
                    doc = [img_info, objects]
                    self.all_docs.append(doc)
                    if args.data_debug:
                        if len(self.all_docs) > 10000:
                            self.debug_flag = True
                            break
                if self.debug_flag:
                    break
        else:
            raise ValueError("on_memory = False Not supported yet!")

        self.num_docs = len(self.all_docs)
        logging.info(
            "deleted {} lines from pretrain corpus from flickr test/val".format(fk_count)
        )
        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )
        print('image load info:', self.img_info_counter)
        del(self.img_info_counter)
        if args.display_time:
            self.tag_time = 0.0
            self.tokenize_time = 0.0
            self.img_time = 0.0
            self.convert_time = 0.0

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.num_docs

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def __getitem__(self, index):
        self.sample_counter += 1
        image_info, t1 = self.all_docs[index]
        ds_name, img_feat = self.get_img_feature(image_info)
        lang, caption = self.concate_captions(t1, ds_name)
        if self.sample_counter == 1:
            logging.info('Input Example')
            logging.info('dataset: {}, image id: {}'.format(ds_name, image_info))
            logging.info('language: {}, caption: {}'.format(lang, caption))
        return (img_feat, caption)
    
    def concate_captions(self, line, dataset_name=None):
        if isinstance(line, dict):
            lang = random.choice(list(line.keys()))
            return lang, self.sep_token[lang].join(line[lang])
        elif isinstance(line, list):
            lang = random.choice(list(self.label_map[dataset_name].keys()))
            return lang, self.sep_token[lang].join([self.label_map[dataset_name][lang][t] for t in line])
        else:
            raise NotImplementedError

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            # split, img_id = img_id.split('_')
            # img_fn = 'COCO_{}_{:012}.jpg'.format(split, int(img_id))
            # img_path = os.path.join(self.image_path[datasetname], split, img_fn)
            img_path = os.path.join(self.image_path[datasetname], img_id)
        elif datasetname == 'oi':
            split_name = 'train_{}'.format(img_id[0])
            img_path = os.path.join(self.image_path[datasetname], split_name, img_id)
        elif datasetname == 'vg':
            img_path = os.path.join(self.image_path[datasetname], img_id)
        else:
            return None
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return datasetname, img

    def check_img_exists(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            # split, img_id = img_id.split('_')
            # img_fn = 'COCO_{}_{:012}.jpg'.format(split, int(img_id))
            # img_path = os.path.join(self.image_path[datasetname], split, img_fn)
            img_path = os.path.join(self.image_path[datasetname], img_id)
        elif datasetname == 'oi':
            split_name = 'train_{}'.format(img_id[0])
            img_path = os.path.join(self.image_path[datasetname], split_name, img_id)
        elif datasetname == 'vg':
            img_path = os.path.join(self.image_path[datasetname], img_id)
        else:
            return False
        return os.path.exists(img_path)
        #     return False, 'image not exist'
        # else:
        #     try:
        #         img = Image.open(img_path).convert('RGB')
        #         return True, 'valid image'
        #     except:
        #         return False, 'not valid image'

    def load_json_corpus(self, corpus_info):
        assert corpus_info['format'] == 'json'
        corpus_file = corpus_info['path']
        i = 0
        if not corpus_file.startswith('/'):
            corpus_file = os.path.join(self.root, corpus_file)
        json_lines = json.load(open(corpus_file, 'r'))
        for line in json_lines:
            img_id = line[0]+'.jpg'
            yield img_id, line[1]

    def load_data_corpus(self, corpus_info):
        assert corpus_info['format'] == 'data'
        corpus_path = corpus_info['path']
        if not corpus_path.startswith('/'):
            corpus_path = os.path.join(self.root, corpus_path)
        for data_split in os.listdir(corpus_path):
            split_file_path = os.path.join(corpus_path, data_split)
            with open(split_file_path, 'r') as rf:
                for line in rf:
                    info = json.loads(line)
                    img_id = os.path.split(info['image'])[1]
                    lang2objects = {}
                    for lang in self.languages:
                        full_objs = []
                        for obj in info['elems']:
                            if isinstance(obj['caption'], list):
                                tmp_tag = obj['caption'][0]
                            else:
                                tmp_tag = obj['caption']
                            full_objs.append(tmp_tag[lang])
                        lang2objects[lang] = full_objs
                    yield img_id, lang2objects





class OscarJsonDatasetImgTrans(Dataset):
    # Json based image-text dataset
    def __init__(self, root, config, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8",  on_memory=True, ds_names=None, transform=None,
                 **kwargs):
        self.cfg = config
        self.root = root
        self.tokenizer = tokenizer
        global cls_token
        if isinstance(self.tokenizer, XLMTokenizer):
            cls_token = self.tokenizer.bos_token
        else:
            cls_token = self.tokenizer.cls_token
        self.seq_len = seq_len
        self.transform = transform
        self.on_memory = on_memory
        global mask_prob
        mask_prob = args.mask_prob
        self.languages = set(args.languages)
        self.sep_token = {'zh': '，'}
        logging.info('mask with {} probabilitiy'.format(mask_prob))

        if ds_names is None:
            self.only_image = False
            self.datasets_names = self.cfg['corpus'].split('_')
        else:
            self.only_image = True
            self.datasets_names = ds_names.split('_')

        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_path = self.cfg['image_path']
        for k,v in self.image_path.items():
            if v.startswith('/'):
                # absolute path
                self.image_path[k] = v
            else:
                # relative path
                self.image_path[k] = os.path.join(self.root, v)

        self.encoding = encoding
        self.dataset_with_splits = ['coco']
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # load samples into memory
        fk_count = 0
        # for checking the image loading information!
        self.img_info_counter = Counter()
        if on_memory:
            self.all_docs = []
            self.tuple_mask_map = []
            self.all_qa_docs = []
            self.all_qa_ans = []
            self.imgid2labels = {}
            self.corpus_lines = 0
            self.wk_count = 0
            max_tokens = 0
            self.debug_flag = False
            # tmp_tag2id = Counter()
            for dataset_name, corpus_info in self.cfg['corpus_file'].items():
                if dataset_name not in self.datasets_names:
                    continue
                corpus_file = corpus_info['path']
                corpus_format = corpus_info['format']
                if 'url2index' in corpus_info:
                    tmp_url2index = json.load(open(corpus_info['url2index'], 'r'))
                    url_flag = True
                else:
                    url_flag = False
                if corpus_format == 'json':
                    corpus_iter = self.load_json_corpus(corpus_info)
                elif corpus_format == 'data':
                    corpus_iter = self.load_data_corpus(corpus_info)
                else:
                    raise NotImplementedError
                for line in tqdm(corpus_iter, desc='loading the {} dataset'.format(dataset_name)):
                    img_id, captions = line
                    if url_flag:
                        if img_id not in tmp_url2index:
                            continue
                        img_id = tmp_url2index[img_id]
                    if isinstance(img_id, str):
                        if img_id.endswith('.jpg'):
                            img_id = img_id[:-4]
                    img_info = '{}|{}'.format(dataset_name, img_id)
                    if not self.check_img_exists(img_info):
                        continue
                    self.img_info_counter.update([dataset_name])
                    doc = [img_info, captions]
                    self.all_docs.append(doc)
                    if args.data_debug:
                        if len(self.all_docs) > 400000:
                            self.debug_flag = True
                            break
                if self.debug_flag:
                    break
        else:
            raise ValueError("on_memory = False Not supported yet!")

        self.num_docs = len(self.all_docs)
        logging.info(
            "deleted {} lines from pretrain corpus from flickr test/val".format(fk_count)
        )
        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )
        print('image load info:', self.img_info_counter)
        del(self.img_info_counter)
        if args.display_time:
            self.tag_time = 0.0
            self.tokenize_time = 0.0
            self.img_time = 0.0
            self.convert_time = 0.0

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.num_docs

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def __getitem__(self, index):
        self.sample_counter += 1
        image_info, t1 = self.all_docs[index]
        ds_name, img_feat = self.get_img_feature(image_info)
        lang, caption = self.concate_captions(t1)
        if self.sample_counter == 1:
            logging.info('Input Example')
            logging.info('dataset: {}, image id: {}'.format(ds_name, image_info))
            logging.info('language: {}, caption: {}'.format(lang, caption))
        return (img_feat, caption)
    
    def concate_captions(self, line):
        lang = random.choice(list(line.keys()))
        return lang, line[lang]

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            cc_head, split, img_id = img_id.split('_')
            img_fn = 'COCO_{}_{:012}.jpg'.format(split, int(img_id))
            img_path = os.path.join(self.image_path[datasetname], split, img_fn)
        elif datasetname == 'cc':
            img_fn = '{:07}.jpg'.format(int(img_id))
            img_path = os.path.join(self.image_path[datasetname], img_fn)
        elif datasetname == 'vg':
            img_fn = '{}.jpg'.format(img_id)
            img_path = os.path.join(self.image_path[datasetname], img_fn)
        else:
            return None
        img = Image.open(img_path)# .convert('RGB')
        try:
            img = self.transform(img)
        except:
            img = self.transform(img.convert('RGB'))
        return datasetname, img

    def check_img_exists(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            cc_head, split, img_id = img_id.split('_')
            img_fn = 'COCO_{}_{:012}.jpg'.format(split, int(img_id))
            img_path = os.path.join(self.image_path[datasetname], split, img_fn)
        elif datasetname == 'cc':
            img_fn = '{:07}.jpg'.format(int(img_id))
            img_path = os.path.join(self.image_path[datasetname], img_fn)
        elif datasetname == 'vg':
            img_fn = '{}.jpg'.format(img_id)
            img_path = os.path.join(self.image_path[datasetname], img_fn)
        else:
            return None
        return os.path.exists(img_path)
        # if os.path.exists(img_path):
        #     return Image.open(img_path).mode=='RGB'
        # else:
        #     return False
        #     return False, 'image not exist'
        # else:
        #     try:
        #         img = Image.open(img_path).convert('RGB')
        #         return True, 'valid image'
        #     except:
        #         return False, 'not valid image'

    def load_json_corpus(self, corpus_info):
        assert corpus_info['format'] == 'json'
        corpus_file = corpus_info['path']
        i = 0
        if not corpus_file.startswith('/'):
            corpus_file = os.path.join(self.root, corpus_file)
        json_lines = json.load(open(corpus_file, 'r'))
        for line in json_lines:
            img_id = line[0]+'.jpg'
            yield img_id, line[1]

    def load_data_corpus(self, corpus_info):
        assert corpus_info['format'] == 'data'
        corpus_path = corpus_info['path']
        if not corpus_path.startswith('/'):
            corpus_path = os.path.join(self.root, corpus_path)
        for data_split in os.listdir(corpus_path):
            split_file_path = os.path.join(corpus_path, data_split)
            with open(split_file_path, 'r') as rf:
                for line in rf:
                    info = json.loads(line)
                    if 'url' in info:
                        img_id = info['url']
                    else:
                        img_id = os.path.split(info['image'])[1]
                    if isinstance(info['caption'], list):
                        for cap in info['caption']:
                            well_caps = {k:v for k,v in cap.items() if k in self.languages}
                            if len(well_caps) == 0:
                                continue
                            yield img_id, well_caps
                    else:
                        well_caps = {k:v for k,v in info['caption'].items() if k in self.languages}
                        if len(well_caps) == 0:
                            continue
                        yield img_id, well_caps


class ArrowJsonDatasetImgTrans(Dataset):
    # Json based image-text dataset and load images from arrow
    def __init__(self, root, config, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8",  on_memory=True, ds_names=None, transform=None,
                 **kwargs):
        self.cfg = config
        self.root = root
        self.tokenizer = tokenizer
        global cls_token
        if isinstance(self.tokenizer, XLMTokenizer):
            cls_token = self.tokenizer.bos_token
        else:
            cls_token = self.tokenizer.cls_token
        self.seq_len = seq_len
        self.transform = transform
        self.on_memory = on_memory
        global mask_prob
        mask_prob = args.mask_prob
        self.languages = set(args.languages)
        self.sep_token = {'zh': '，'}
        logging.info('mask with {} probabilitiy'.format(mask_prob))

        if ds_names is None:
            self.only_image = False
            self.datasets_names = self.cfg['corpus'].split('_')
        else:
            self.only_image = True
            self.datasets_names = ds_names.split('_')

        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_path = self.cfg['image_path']
        for k,v in self.image_path.items():
            if v.startswith('/'):
                # absolute path
                self.image_path[k] = v
            else:
                # relative path
                self.image_path[k] = os.path.join(self.root, v)

        self.encoding = encoding
        self.dataset_with_splits = ['coco', 'cc']
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # load samples into memory
        fk_count = 0
        # for checking the image loading information!
        self.img_info_counter = Counter()
        if on_memory:
            self.all_docs = []
            self.tuple_mask_map = []
            self.all_qa_docs = []
            self.all_qa_ans = []
            self.imgid2labels = {}
            self.arrow_table = {}
            self.arrow_indexmap = {}
            self.corpus_lines = 0
            self.wk_count = 0
            max_tokens = 0
            self.debug_flag = False
            # tmp_tag2id = Counter()
            for dataset_name, corpus_info in self.cfg['corpus_file'].items():
                # processing the text part data
                if dataset_name not in self.datasets_names:
                    continue
                corpus_file = corpus_info['path']
                corpus_format = corpus_info['format']
                if 'url2index' in corpus_info:
                    tmp_url2index = json.load(open(corpus_info['url2index'], 'r'))
                    url_flag = True
                else:
                    url_flag = False
                if corpus_format == 'json':
                    corpus_iter = self.load_json_corpus(corpus_info)
                elif corpus_format == 'data':
                    corpus_iter = self.load_data_corpus(corpus_info)
                else:
                    raise NotImplementedError
                # processing the image part data
                self.load_arrow_data(dataset_name)

                for line in tqdm(corpus_iter, desc='loading the {} dataset'.format(dataset_name)):
                    img_id, captions = line
                    if url_flag:
                        if img_id not in tmp_url2index:
                            continue
                        img_id = tmp_url2index[img_id]
                    if isinstance(img_id, str):
                        if img_id.endswith('.jpg'):
                            img_id = img_id[:-4]
                    img_info = '{}|{}'.format(dataset_name, img_id)
                    if not self.check_img_exists(img_info):
                        continue
                    self.img_info_counter.update([dataset_name])
                    doc = [img_info, captions]
                    self.all_docs.append(doc)
                    if args.data_debug:
                        if len(self.all_docs) > 400000:
                            self.debug_flag = True
                            break
                if self.debug_flag:
                    break
        else:
            raise ValueError("on_memory = False Not supported yet!")

        self.num_docs = len(self.all_docs)
        logging.info(
            "deleted {} lines from pretrain corpus from flickr test/val".format(fk_count)
        )
        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )
        print('image load info:', self.img_info_counter)
        del(self.img_info_counter)
        if args.display_time:
            self.tag_time = 0.0
            self.tokenize_time = 0.0
            self.img_time = 0.0
            self.convert_time = 0.0

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.num_docs

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def __getitem__(self, index):
        self.sample_counter += 1
        image_info, t1 = self.all_docs[index]
        ds_name, img_feat = self.get_img_feature(image_info)
        lang, caption = self.concate_captions(t1)
        if self.sample_counter == 1:
            logging.info('Input Example')
            logging.info('dataset: {}, image id: {}'.format(ds_name, image_info))
            logging.info('language: {}, caption: {}'.format(lang, caption))
        return (img_feat, caption)
    
    def concate_captions(self, line):
        lang = random.choice(list(line.keys()))
        return lang, line[lang]

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            cc_head, split, img_id = img_id.split('_')
            img_fn = 'COCO_{}_{:012}'.format(split, int(img_id))
            img_path = BytesIO(self.arrow_table['coco'][split]['image'][self.arrow_indexmap['coco'][split][img_fn]].as_py())
        elif datasetname == 'cc':
            img_fn = '{:07}'.format(int(img_id))
            split = img_fn[:2]
            img_path = BytesIO(self.arrow_table['cc'][split]['image'][self.arrow_indexmap['cc'][split][img_fn]].as_py())
        elif datasetname == 'vg':
            img_path = BytesIO(self.arrow_table['vg']['image'][self.arrow_indexmap['vg'][img_id]].as_py())
        else:
            return None
        img = Image.open(img_path)# .convert('RGB')
        try:
            img = self.transform(img)
        except:
            img = self.transform(img.convert('RGB'))
        return datasetname, img

    def load_arrow_data(self, ds_name):
        if ds_name not in self.dataset_with_splits:
            pa_path = self.image_path[ds_name]
            current_table = pa.ipc.RecordBatchFileReader(pa.memory_map(pa_path, 'r')).read_all()
            pa_id2index = {current_table['index'][i].as_py():i for i in range(len(current_table))}
            self.arrow_table[ds_name] = current_table
            self.arrow_indexmap[ds_name] = pa_id2index
        else:
            # arrow with split like cc and coco
            if ds_name == 'cc':
                self.arrow_table[ds_name] = {}
                self.arrow_indexmap[ds_name] = {}
                base_path = os.path.splitext(self.image_path[ds_name])[0]
                for i in range(34):
                    pa_path = '{}_{}.arrow'.format(base_path, i)
                    current_table = pa.ipc.RecordBatchFileReader(pa.memory_map(pa_path, 'r')).read_all()
                    pa_id2index = {current_table['index'][i].as_py():i for i in range(len(current_table))}
                    split_name = '{:02}'.format(i)
                    self.arrow_table[ds_name][split_name] = current_table
                    self.arrow_indexmap[ds_name][split_name] = pa_id2index
            elif ds_name == 'coco':
                self.arrow_table[ds_name] = {}
                self.arrow_indexmap[ds_name] = {}
                base_path = os.path.splitext(self.image_path[ds_name])[0]
                for i in ['train2014', 'val2014']:
                    pa_path = '{}_{}.arrow'.format(base_path, i)
                    current_table = pa.ipc.RecordBatchFileReader(pa.memory_map(pa_path, 'r')).read_all()
                    pa_id2index = {current_table['index'][i].as_py():i for i in range(len(current_table))}
                    split_name = i
                    self.arrow_table[ds_name][split_name] = current_table
                    self.arrow_indexmap[ds_name][split_name] = pa_id2index
            else:
                raise NotImplementedError


    
    def check_img_exists(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            cc_head, split, img_id = img_id.split('_')
            img_fn = 'COCO_{}_{:012}'.format(split, int(img_id))
            return img_fn in self.arrow_indexmap['coco'][split]
        elif datasetname == 'cc':
            img_fn = '{:07}'.format(int(img_id))
            split = img_fn[:2]
            # if split not in self.arrow_indexmap['cc']:
            #     return False
            return img_fn in self.arrow_indexmap['cc'][split]
        elif datasetname == 'vg':
            return img_id in self.arrow_indexmap['vg']
        else:
            return False
        return os.path.exists(img_path)
        # if os.path.exists(img_path):
        #     return Image.open(img_path).mode=='RGB'
        # else:
        #     return False
        #     return False, 'image not exist'
        # else:
        #     try:
        #         img = Image.open(img_path).convert('RGB')
        #         return True, 'valid image'
        #     except:
        #         return False, 'not valid image'

    def load_json_corpus(self, corpus_info):
        assert corpus_info['format'] == 'json'
        corpus_file = corpus_info['path']
        i = 0
        if not corpus_file.startswith('/'):
            corpus_file = os.path.join(self.root, corpus_file)
        json_lines = json.load(open(corpus_file, 'r'))
        for line in json_lines:
            img_id = line[0]+'.jpg'
            yield img_id, line[1]

    def load_data_corpus(self, corpus_info):
        assert corpus_info['format'] == 'data'
        corpus_path = corpus_info['path']
        if not corpus_path.startswith('/'):
            corpus_path = os.path.join(self.root, corpus_path)
        for data_split in os.listdir(corpus_path):
            split_file_path = os.path.join(corpus_path, data_split)
            with open(split_file_path, 'r') as rf:
                for line in rf:
                    info = json.loads(line)
                    if 'url' in info:
                        img_id = info['url']
                    else:
                        img_id = os.path.split(info['image'])[1]
                    if isinstance(info['caption'], list):
                        for cap in info['caption']:
                            well_caps = {k:v for k,v in cap.items() if k in self.languages}
                            if len(well_caps) == 0:
                                continue
                            yield img_id, well_caps
                    else:
                        well_caps = {k:v for k,v in info['caption'].items() if k in self.languages}
                        if len(well_caps) == 0:
                            continue
                        yield img_id, well_caps


class LMDBJsonDatasetImgTrans(Dataset):
    # Json based image-text dataset and load images from LMDB
    def __init__(self, root, config, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8",  on_memory=True, ds_names=None, transform=None,
                 **kwargs):
        self.cfg = config
        self.root = root
        self.tokenizer = tokenizer
        global cls_token
        if isinstance(self.tokenizer, XLMTokenizer):
            cls_token = self.tokenizer.bos_token
        else:
            cls_token = self.tokenizer.cls_token
        self.seq_len = seq_len
        self.transform = transform
        self.on_memory = on_memory
        global mask_prob
        mask_prob = args.mask_prob
        self.languages = set(args.languages)
        self.sep_token = {'zh': '，'}
        logging.info('mask with {} probabilitiy'.format(mask_prob))

        if ds_names is None:
            self.only_image = False
            self.datasets_names = self.cfg['corpus'].split('_')
        else:
            self.only_image = True
            self.datasets_names = ds_names.split('_')

        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_path = self.cfg['image_path']
        for k,v in self.image_path.items():
            if v.startswith('/'):
                # absolute path
                self.image_path[k] = v
            else:
                # relative path
                self.image_path[k] = os.path.join(self.root, v)

        self.encoding = encoding
        self.dataset_with_splits = ['coco', 'cc']
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # load samples into memory
        fk_count = 0
        # for checking the image loading information!
        self.img_info_counter = Counter()
        if on_memory:
            self.all_docs = []
            self.tuple_mask_map = []
            self.all_qa_docs = []
            self.all_qa_ans = []
            self.imgid2labels = {}
            self.lmdb_env = {}
            self.lmdb_txn = {}
            self.lmdb_keys = {}
            self.corpus_lines = 0
            self.wk_count = 0
            max_tokens = 0
            self.debug_flag = False
            # tmp_tag2id = Counter()
            for dataset_name, corpus_info in self.cfg['corpus_file'].items():
                # processing the text part data
                if dataset_name not in self.datasets_names:
                    continue
                corpus_file = corpus_info['path']
                corpus_format = corpus_info['format']
                if 'url2index' in corpus_info:
                    tmp_url2index = json.load(open(corpus_info['url2index'], 'r'))
                    url_flag = True
                else:
                    url_flag = False
                if corpus_format == 'json':
                    corpus_iter = self.load_json_corpus(corpus_info)
                elif corpus_format == 'data':
                    corpus_iter = self.load_data_corpus(corpus_info)
                else:
                    raise NotImplementedError
                # processing the image part data
                self.load_lmdb_data(dataset_name)

                for line in tqdm(corpus_iter, desc='loading the {} dataset'.format(dataset_name)):
                    img_id, captions = line
                    if url_flag:
                        if img_id not in tmp_url2index:
                            continue
                        img_id = tmp_url2index[img_id]
                    if isinstance(img_id, str):
                        if img_id.endswith('.jpg'):
                            img_id = img_id[:-4]
                    img_info = '{}|{}'.format(dataset_name, img_id)
                    if not self.check_img_exists(img_info):
                        continue
                    self.img_info_counter.update([dataset_name])
                    doc = [img_info, captions]
                    self.all_docs.append(doc)
                    if args.data_debug:
                        if len(self.all_docs) > 400000:
                            self.debug_flag = True
                            break
                if self.debug_flag:
                    break
        else:
            raise ValueError("on_memory = False Not supported yet!")

        self.num_docs = len(self.all_docs)
        logging.info(
            "deleted {} lines from pretrain corpus from flickr test/val".format(fk_count)
        )
        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )
        print('image load info:', self.img_info_counter)
        del self.lmdb_keys
        del(self.img_info_counter)
        if args.display_time:
            self.tag_time = 0.0
            self.tokenize_time = 0.0
            self.img_time = 0.0
            self.convert_time = 0.0

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.num_docs

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def __getitem__(self, index):
        self.sample_counter += 1
        image_info, t1 = self.all_docs[index]
        ds_name, img_feat = self.get_img_feature(image_info)
        lang, caption = self.concate_captions(t1)
        if self.sample_counter == 1:
            logging.info('Input Example')
            logging.info('dataset: {}, image id: {}'.format(ds_name, image_info))
            logging.info('language: {}, caption: {}'.format(lang, caption))
        return (img_feat, caption)
    
    def concate_captions(self, line):
        lang = random.choice(list(line.keys()))
        return lang, line[lang]

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            cc_head, split, img_id = img_id.split('_')
            img_fn = 'COCO_{}_{:012}'.format(split, int(img_id))
            img_path = BytesIO(self.lmdb_txn[datasetname][split].get(img_fn.encode('utf-8')))
        elif datasetname == 'cc':
            img_fn = '{:07}'.format(int(img_id))
            split = img_fn[:2]
            img_path = BytesIO(self.lmdb_txn[datasetname][split].get(img_fn.encode('utf-8')))
        elif datasetname == 'vg':
            img_path = BytesIO(self.lmdb_txn[datasetname].get(img_id.encode('utf-8')))
        else:
            return None
        img = Image.open(img_path)# .convert('RGB')
        try:
            img = self.transform(img)
        except:
            img = self.transform(img.convert('RGB'))
        return datasetname, img

    def load_lmdb_data(self, ds_name):
        if ds_name not in self.dataset_with_splits:
            lmdb_path = self.image_path[ds_name]
            tmp_env = lmdb.open(lmdb_path, readonly=True, readahead=False)
            tmp_txn = tmp_env.begin(buffers=True)
            self.lmdb_env[ds_name] = tmp_env
            self.lmdb_txn[ds_name] = tmp_txn
            valid_keys = []
            for k,v in tmp_txn.cursor():
                valid_keys.append(str(BytesIO(k).read(), encoding='utf-8'))
            self.lmdb_keys[ds_name] = set(valid_keys)
        else:
            # arrow with split like cc and coco
            if ds_name == 'cc':
                self.lmdb_env[ds_name] = {}
                self.lmdb_txn[ds_name] = {}
                self.lmdb_keys[ds_name] = {}
                base_path = self.image_path[ds_name]
                for i in range(34):
                    lmdb_path = '{}_{}'.format(base_path, i)
                    tmp_env = lmdb.open(lmdb_path, readonly=True, readahead=False)
                    tmp_txn = tmp_env.begin(buffers=True)
                    split_name = '{:02}'.format(i)
                    self.lmdb_env[ds_name][split_name] = tmp_env
                    self.lmdb_txn[ds_name][split_name] = tmp_txn
                    valid_keys = []
                    for k,v in tmp_txn.cursor():
                        valid_keys.append(str(BytesIO(k).read(), encoding='utf-8'))
                    self.lmdb_keys[ds_name][split_name] = set(valid_keys)
            elif ds_name == 'coco':
                self.lmdb_env[ds_name] = {}
                self.lmdb_txn[ds_name] = {}
                self.lmdb_keys[ds_name] = {}
                base_path = self.image_path[ds_name]
                for i in ['train2014', 'val2014']:
                    print('processing the coco {} split'.format(i))
                    lmdb_path = '{}_{}'.format(base_path, i)
                    tmp_env = lmdb.open(lmdb_path, readonly=True, readahead=False)
                    tmp_txn = tmp_env.begin(buffers=True)
                    split_name = i
                    self.lmdb_env[ds_name][split_name] = tmp_env
                    self.lmdb_txn[ds_name][split_name] = tmp_txn
                    valid_keys = []
                    for k,v in tmp_txn.cursor():
                        valid_keys.append(str(BytesIO(k).read(), encoding='utf-8'))
                    self.lmdb_keys[ds_name][split_name] = set(valid_keys)
            else:
                raise NotImplementedError
    
    def check_img_exists(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname == 'coco':
            cc_head, split, img_id = img_id.split('_')
            img_fn = 'COCO_{}_{:012}'.format(split, int(img_id))
            return img_fn in self.lmdb_keys['coco'][split]
        elif datasetname == 'cc':
            img_fn = '{:07}'.format(int(img_id))
            split = img_fn[:2]
            return img_fn in self.lmdb_keys['cc'][split]
        elif datasetname == 'vg':
            return img_id in self.lmdb_keys['vg']
        else:
            return None
        return os.path.exists(img_path)
        # if os.path.exists(img_path):
        #     return Image.open(img_path).mode=='RGB'
        # else:
        #     return False
        #     return False, 'image not exist'
        # else:
        #     try:
        #         img = Image.open(img_path).convert('RGB')
        #         return True, 'valid image'
        #     except:
        #         return False, 'not valid image'

    def load_json_corpus(self, corpus_info):
        assert corpus_info['format'] == 'json'
        corpus_file = corpus_info['path']
        i = 0
        if not corpus_file.startswith('/'):
            corpus_file = os.path.join(self.root, corpus_file)
        json_lines = json.load(open(corpus_file, 'r'))
        for line in json_lines:
            img_id = line[0]+'.jpg'
            yield img_id, line[1]

    def load_data_corpus(self, corpus_info):
        assert corpus_info['format'] == 'data'
        corpus_path = corpus_info['path']
        if not corpus_path.startswith('/'):
            corpus_path = os.path.join(self.root, corpus_path)
        for data_split in os.listdir(corpus_path):
            split_file_path = os.path.join(corpus_path, data_split)
            with open(split_file_path, 'r') as rf:
                for line in rf:
                    info = json.loads(line)
                    if 'url' in info:
                        img_id = info['url']
                    else:
                        img_id = os.path.split(info['image'])[1]
                    if isinstance(info['caption'], list):
                        for cap in info['caption']:
                            well_caps = {k:v for k,v in cap.items() if k in self.languages}
                            if len(well_caps) == 0:
                                continue
                            yield img_id, well_caps
                    else:
                        well_caps = {k:v for k,v in info['caption'].items() if k in self.languages}
                        if len(well_caps) == 0:
                            continue
                        yield img_id, well_caps