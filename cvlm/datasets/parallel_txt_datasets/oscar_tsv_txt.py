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

phrase_vocab_size = None
theme_vocab_size = None
mask_prob = 0.15
phrase_type_id = 0
concept_id2key = {}
all_concept_ids = []
only_multi_phrase = False
cls_token = '[CLS]'

class ParallelTxtDataset(Dataset):
    def __init__(self, root, config, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8", corpus_lines=None, on_memory=True, ds_names=None, transform=None,
                 **kwargs):
        self.cfg = config
        self.root = root
        self.tokenizer = tokenizer
        global cls_token
        if isinstance(self.tokenizer, XLMTokenizer):
            cls_token = self.tokenizer.bos_token
            lang2id = self.tokenizer.lang2id
        else:
            cls_token = self.tokenizer.cls_token
            lang2id = None
        self.seq_len = seq_len
        self.on_memory = on_memory
        global mask_prob
        mask_prob = args.mask_prob
        logging.info('mask with {} probabilitiy'.format(mask_prob))
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus

        if ds_names is None:
            self.datasets_names = self.cfg['corpus'].split('_')
        else:
            self.datasets_names = ds_names.split('_')

        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))

        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line
        
        if on_memory:
            self.all_docs = []
            self.tuple_mask_map = []
            self.all_qa_docs = []
            self.all_qa_ans = []
            self.imgid2labels = {}
            self.corpus_lines = 0
            self.wk_count = 0
            max_tokens = 0
            # tmp_tag2id = Counter()
            for dataset_name, dataset_info in self.cfg['corpus_file'].items():
                if dataset_name not in self.datasets_names:
                    continue
                for lang_pair, caption_file in dataset_info.items():
                    src_lang, tar_lang = lang_pair.split('-')
                    src_lang_id = lang2id[src_lang] if lang2id is not None else 0
                    tar_lang_id = lang2id[tar_lang] if lang2id is not None else 0
                    src_file, tar_file = caption_file
                    if not src_file.startswith('/'):
                        src_file = os.path.join(self.root, src_file)
                    if not tar_file.startswith('/'):
                        tar_file = os.path.join(self.root, tar_file)
                    with open(src_file, 'r') as src_rf, open(tar_file, 'r') as tar_rf:
                        for line in tqdm(zip(src_rf, tar_rf), desc='loading the {} pair in {}'.format(lang_pair, dataset_name)):
                            self.all_docs.append([line[0], line[1], src_lang_id, tar_lang_id])
                
            self.num_docs = len(self.all_docs)
        # load samples later lazily from disk
        else:
            raise ValueError("on_memory = False Not supported yet!")

        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
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
        src_t, tar_t, src_lang_id, tar_lang_id = self.all_docs[index]
        return (src_t.strip(), tar_t.strip(), torch.ones(1, dtype=torch.long)*src_lang_id, torch.ones(1, dtype=torch.long)*tar_lang_id)

    
    def __getitem2__(self, item):
        time_0 = time.time()
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                raise ValueError("on_memory = False Not supported yet!")

        # img_id, t1, t2, is_next_label, is_img_match, qa_ans, p_c, doc_idx = self.random_sent(item)
        image_info, t1 = self.all_docs[item]
        # t2 = self.get_img_tags(image_info)
        time_1 = time.time()

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        # tokens_a = [t for t in t1]

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                   img_id=image_info)

        time_2 = time.time()
        # get image feature
        img_feat = self.get_img_feature(image_info)

        time_3 = time.time()
        cur_features = convert_example_to_features(self.args, cur_example,
                                                   self.seq_len, self.tokenizer)

        time_4 = time.time()
        if self.args.display_time:
            self.tag_time += time_1 - time_0
            self.tokenize_time += time_2 - time_1
            self.img_time += time_3 - time_2
            self.convert_time += time_4 - time_3
            if self.sample_counter % 10 == 0:
                print('average tag time {:.3f}, tokenize time {:.3f}, image time {:.3f}, convert time {:.3f}'.format(
                    self.tag_time / self.sample_counter, self.tokenize_time / self.sample_counter, self.img_time / self.sample_counter, self.convert_time / self.sample_counter
                ))

        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(cur_features.input_ids_a, dtype=torch.long),
                torch.tensor(cur_features.input_mask_a, dtype=torch.long),
                torch.tensor(cur_features.segment_ids_a, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids_a, dtype=torch.long),
                item)
        else:
            return img_feat, (
                torch.tensor(cur_features.input_ids, dtype=torch.long),
                torch.tensor(cur_features.input_mask, dtype=torch.long),
                torch.tensor(cur_features.segment_ids, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
                torch.tensor(cur_features.is_next),
                torch.tensor(cur_features.is_img_match)
                ), item
        # return cur_tensors

    def get_img_tags(self, image_info):
        dataset_name, image_id = image_info.split('|')
        meta_info = self.od_reader[dataset_name].read_many([image_id])[0]
        return ' '.join(str(meta_info, encoding='utf-8').split(';'))

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        img_id, t1, t2, qa_ans, p_c = self.get_corpus_line(index)
        # qa_ans = None if not a QA-pair
        doc_idx = index
        rand_dice = random.random()

        if qa_ans is not None:
            # as a qa pair
            if rand_dice > 0.5:
                # wrong qa pair
                rand_qa_ans = self.get_random_qa()
                if rand_qa_ans == qa_ans:
                    label = 0
                else:
                    qa_ans = rand_qa_ans
                    label = 1
            else:
                # true qa pair
                label = 0
            random_img_id = img_id
        elif rand_dice >= 0: # changed to >=0 here to make it always true
            label = -1
            random_img_id = img_id
        elif rand_dice > self.args.texta_false_prob and t2 != "":
            # wrong qa triplets
            random_img_id, t2, n_v_c = self.get_random_line()
            if self.args.change_theme:
                v_c = n_v_c
            label = -1
        else:
            # wrong retrieval triplets
            random_img_id, t1, p_c, doc_idx = self.get_random_texta()
            # args.num_contrast_classes = 3 if args.texta_false_prob<0.5 and (args.texta_false_prob>0 or not args.use_b) else 2
            label = -1
            # label = self.args.num_contrast_classes-1

        img_match_label = 0
        if img_id != random_img_id: img_match_label = 1

        assert len(t1) > 0
        assert len(t2) > 0 or not self.args.use_b
        return img_id, t1, t2, label, img_match_label, qa_ans, p_c, doc_idx

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
            img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
            only_img_id = img_id.split('_')
            only_img_id = only_img_id[0]+'_'+only_img_id[-1]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            self.current_img = img_id

            # get extra concepts
            # v_c = self.img2theme[only_img_id] # visual theme concepts
            qa_ans = self.all_qa_ans[item]
            p_c = self.all_docs[sample["doc_id"]][-1] # textual phrase concepts

            assert t1 != ""
            if self.args.use_b or 'qa' in self.all_docs[sample["doc_id"]][1].split('_'):
                assert t2 != ""
            else:
                t2 = ""
            return img_id, t1, t2, qa_ans, p_c
        else:
            raise ValueError("on_memory = False Not supported yet!")

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        if self.on_memory:
            if self.textb_sample_mode in [0, 1]:
                # sample from all docs
                for _ in range(10):
                    rand_doc_idx = random.randrange(0, len(self.all_docs))
                    img_id = self.all_docs[rand_doc_idx][0].split('|')[0]
                    # check if our picked random line is really from another image like we want it to be
                    if img_id != self.current_img:
                        break
                rand_doc = self.all_docs[rand_doc_idx]
            else:
                # sample from all qa docs
                for _ in range(10):
                    rand_doc_idx = random.randrange(0, len(self.all_qa_docs))
                    # check if our picked random line is really from another doc like we want it to be % no need to be different image here
                    if self.all_qa_docs[rand_doc_idx]["doc_id"] != self.current_doc:
                        break
                rand_doc = self.all_qa_docs[rand_doc_idx]["doc"]
            # img_id = rand_doc[0] # original
            img_id = rand_doc[0].split('|')[0]
            if self.textb_sample_mode == 0:
                # default oscar sample mode
                line = rand_doc[random.randrange(1, len(rand_doc))]
            else:
                # only sample text_b
                line = rand_doc[2]
            only_img_id = img_id.split('_')
            only_img_id = only_img_id[0]+'_'+only_img_id[-1]
            v_c = self.img2theme[only_img_id]
            return img_id, line, v_c
        else:
            raise ValueError("on_memory = False Not supported yet!")

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname in self.dataset_with_splits:
            split, img_id = img_id.split('_')
            img_path = os.path.join(self.image_path[datasetname], split, img_id)
        else:
            img_path = os.path.join(self.image_path[datasetname], img_id)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def check_img_exists(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        datasetname, img_id = image_id.split('|')
        if datasetname in self.dataset_with_splits:
            split, img_id = img_id.split('_')
            img_path = os.path.join(self.image_path[datasetname], split, img_id)
        else:
            img_path = os.path.join(self.image_path[datasetname], img_id)
        return os.path.exists(img_path)
        #     return False, 'image not exist'
        # else:
        #     try:
        #         img = Image.open(img_path).convert('RGB')
        #         return True, 'valid image'
        #     except:
        #         return False, 'not valid image'



class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None,
                 lm_labels=None, img_id=None, is_img_match=None,
                 img_label=None, qa_ans=None, phrase_concept=None,
                 phrase_mask_map=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.lm_labels = lm_labels  # masked words for language model

        self.img_id = img_id
        self.qa_ans = qa_ans


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_a, input_mask_a, segment_ids_a, is_next=None, lm_label_ids_a=None):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a
        self.is_next = is_next
        self.lm_label_ids_a = lm_label_ids_a

def random_word_naive(tokens, tokenizer):
    return tokens, [-1]*len(tokens)

def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    
    if hasattr(tokenizer, 'vocab'):
        tmp_vocab = tokenizer.vocab
    else:
        tmp_vocab = tokenizer.get_vocab()

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        # if prob < 0.15:
        #     prob /= 0.15
        if prob < mask_prob:   # edited here for larger masking probability
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = tokenizer.mask_token

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tmp_vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tmp_vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tmp_vocab[tokenizer.unk_token])
                logging.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(
                        token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def random_phrases(tokenizer, old_phrase_nodes, t1_label, phrase_mask_map):
    phrase_nodes = [n for n in old_phrase_nodes]
    output_label = []
    new_phrase_nodes = []
    already_masked = set()
    for i,t in enumerate(t1_label):
        if t >= 0:
            if i in phrase_mask_map:
                already_masked.update(phrase_mask_map[i])
    # print('test:', [i for i,t in enumerate(t1_label) if t>=0], phrase_mask_map, already_masked)
    for i, phrase in enumerate(phrase_nodes):
        if only_multi_phrase and phrase < 30522:
            continue
        if i in already_masked:
            output_label.append(-1) # can not be recovered concepts
            # phrase_nodes[i] = tokenizer.vocab['[MASK]']
            new_phrase_nodes.append(tokenizer.vocab['[MASK]'])
        else:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    # phrase_nodes[i] = tokenizer.vocab['[MASK]']
                    new_phrase_nodes.append(tokenizer.vocab['[MASK]'])

                # 10% randomly change token to random token
                elif prob < 0.9:
                    new_phrase_nodes.append(all_concept_ids[random.randint(0, len(all_concept_ids)-1)])
                    # phrase_nodes[i] = all_concept_ids[random.randint(0, len(all_concept_ids)-1)]
                else:
                    new_phrase_nodes.append(phrase)
                    # phrase_nodes[i] = random.randint(0, phrase_vocab_size-1)+tokenizer.vocab_size
                output_label.append(concept_id2key[phrase])
            else:
                new_phrase_nodes.append(phrase)
                output_label.append(-1)
    assert len(new_phrase_nodes) == len(output_label)
    return new_phrase_nodes, output_label


def random_theme(theme_nodes, tokenizer):
    output_label = []
    for i, t in enumerate(theme_nodes):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.8:
                theme_nodes[i] = tokenizer.vocab['[MASK]'] - len(tokenizer.vocab) - phrase_vocab_size

            elif prob < 0.9:
                theme_nodes[i] = random.randint(0, theme_vocab_size-1)
            output_label.append(t)
        else:
            output_label.append(-1 - len(tokenizer.vocab) - phrase_vocab_size)
    return theme_nodes, output_label


def random_visual(regions, od_tags, tag2id):
    """
    Masking some random regions for Masked Region task with probabilities as in the VLP papers.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    od_labels = od_tags.lower().split('\t')
    output_label = []
    mask_region_id = []

    # print(od_labels, len(od_labels), regions.shape[0])
    for i in range(regions.shape[0]):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            regions[i, :-6] = 0 # mask region
            output_label.append(tag2id[od_labels[i]] if od_labels[i] in tag2id else -1)
            mask_region_id.append(1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            mask_region_id.append(0)

    return regions, output_label, mask_region_id

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def convert_example_to_features(args, example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param args: parameter settings
    :param img_feat_len: lens of actual img features
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    # extra_concept part
    
    tokens_a = example.tokens_a
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # is_next_type = example.is_next * example.is_img_match # is_img_match = 1 for mismatch images
    # if args.num_contrast_classes == 2 and args.texta_false_prob == 0.5 and is_next_type == 1:
    #     is_next_type = 2 # is_next_type 0: correct pair, 1: wrong text_b, 2: wrong text_a
    # if not args.mask_loss_for_unmatched and is_next_type == 2:
    #     t1_label = [-1]*len(tokens_a)
    # else:
    tokens_a, t1_label = random_word(tokens_a, tokenizer)
        # if not args.mask_loss_for_unmatched and is_next_type == 1:
        #     t2_label = [-1]*len(tokens_b)
        # else:
        #     tokens_b, t2_label = random_word(tokens_b, tokenizer)

    # theme_nodes = [t+tokenizer.vocab_size+phrase_vocab_size for t in theme_nodes]
    # theme_label = [t+tokenizer.vocab_size+phrase_vocab_size for t in theme_label]
    # theme_label = [-1 for p in theme_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    # sequence_a (textual side)
    lm_label_ids_a = ([-1] + t1_label + [-1])

    seq_tokens_a = []
    segment_ids_a = []
    seq_tokens_a.append(cls_token)
    segment_ids_a.append(0)
    for token in tokens_a:
        seq_tokens_a.append(token)
        segment_ids_a.append(0)
    seq_tokens_a.append(tokenizer.sep_token)
    input_ids_a = tokenizer.convert_tokens_to_ids(seq_tokens_a)
    segment_ids_a.append(0)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask_a = [1] * len(input_ids_a)

    # Zero-pad up to the sequence length.
    if hasattr(tokenizer, 'pad_token_id'):
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = tokenizer.vocab[tokenizer.pad_token]
    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(pad_id)
        input_mask_a.append(0)
        segment_ids_a.append(0)
        lm_label_ids_a.append(-1)

    # s = "wrong length as maximal {}, input_ids {}, input_mask {}, segment{}, lm_label{}".format(max_seq_length, len(input_ids), len(input_mask), len(segment_ids), len(lm_label_ids))
    s = 'not valid sequence length, please check'
    assert len(input_ids_a) == max_seq_length, s + 'current length {}'.format(len(input_ids_a))
    assert len(input_mask_a) == max_seq_length, s + 'current length {}'.format(len(input_mask_a))
    assert len(segment_ids_a) == max_seq_length, s + 'current length {}'.format(len(segment_ids_a))
    assert len(lm_label_ids_a) == max_seq_length, s + 'current length {}'.format(len(lm_label_ids_a))

    if example.guid < 1:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens_a: %s" % " ".join([str(x) for x in seq_tokens_a]))
        logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
        logging.info("segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))
        logging.info("LM label seq A: %s " % lm_label_ids_a)
        # logging.info("Is next sentence label: %s " % example.is_next)

    features = InputFeatures(input_ids_a=input_ids_a,
                             input_mask_a=input_mask_a,
                             segment_ids_a=segment_ids_a,
                             lm_label_ids_a=lm_label_ids_a,
                             is_next=None)
    return features


def convert_qa_example_to_features(args, example, max_seq_length, tokenizer,
                                img_feat_len, num_phrases, num_themes):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param args: parameter settings
    :param img_feat_len: lens of actual img features
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    # extra_concept part
    qa_ans = example.qa_ans
    tokens_ans = tokenizer.tokenize(qa_ans)
    phrase_nodes = example.phrase_concept
    phrase_mask_map = example.phrase_mask_map
    
    tokens_a = example.tokens_a
    tokens_b = None
    if example.tokens_b:
        tokens_b = example.tokens_b
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        if len(tokens_b) > args.max_tag_length - 2:
            tokens_b = tokens_b[:(args.max_tag_length)-2]

        _truncate_seq_pair(tokens_a, tokens_ans, max_seq_length-3)
        # if len(tokens_a) > max_seq_length - 2:
        #     tokens_a = tokens_a[:(max_seq_length - 2)]
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # is_next_type = example.is_next * example.is_img_match # is_img_match = 1 for mismatch images
    # if args.num_contrast_classes == 2 and args.texta_false_prob == 0.5 and is_next_type == 1:
    #     is_next_type = 2 # is_next_type 0: correct pair, 1: wrong text_b, 2: wrong text_a
    # if not args.mask_loss_for_unmatched and is_next_type == 2:
    #     t1_label = [-1]*len(tokens_a)
    # else:
    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_ans, ans_label = random_word(tokens_ans, tokenizer)
    if tokens_b:
        tokens_b, t2_label = random_word(tokens_b, tokenizer)
        # if not args.mask_loss_for_unmatched and is_next_type == 1:
        #     t2_label = [-1]*len(tokens_b)
        # else:
        #     tokens_b, t2_label = random_word(tokens_b, tokenizer)

    # else:
    #     theme_mask = [1] * len(theme_nodes) + [0] * (num_themes - len(theme_nodes))
    #     theme_nodes = theme_nodes + [0] * (num_themes - len(theme_nodes))

    if len(phrase_nodes) >= num_phrases+max_seq_length-3-len(tokens_a+tokens_ans):
        phrase_nodes = phrase_nodes[:(num_phrases+max_seq_length-3-len(tokens_a+tokens_ans))]
    phrase_mask = [1] * len(phrase_nodes)
    # else:
    #     phrase_mask = [1] * len(phrase_nodes) + [0] * (num_phrases - len(phrase_nodes))
    #     phrase_nodes = phrase_nodes + [0] * (num_phrases - len(phrase_nodes))

    # input id processing
    phrase_nodes, phrase_label = random_phrases(tokenizer, phrase_nodes, t1_label, phrase_mask_map)
    # theme_nodes, theme_label = random_theme(theme_nodes, tokenizer)

    # phrase_nodes = [p+tokenizer.vocab_size for p in phrase_nodes]
    # phrase_label = [p+tokenizer.vocab_size for p in phrase_label]
    fake_phrase_label = [-1 for p in phrase_label]
    # theme_nodes = [t+tokenizer.vocab_size+phrase_vocab_size for t in theme_nodes]
    # theme_label = [t+tokenizer.vocab_size+phrase_vocab_size for t in theme_label]
    # theme_label = [-1 for p in theme_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    # sequence_a (textual side)
    lm_label_ids_a = ([-1] + t1_label + fake_phrase_label + [-1] + ans_label + [-1])
    lm_label_ids_b = ([-1] + t2_label + [-1])
    phrase_lm_labels = ([-1]*(len(t1_label)+1) + phrase_label + [-1]*(2+len(ans_label)))

    # if tokens_b:
    #     lm_label_ids_b = ([-1] + t1_label + phrase_label + [-1] + t2_label + theme_label + [-1])
    # else:
    #     lm_label_ids = ([-1] + t1_label + phrase_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    seq_tokens_a = []
    segment_ids_a = []
    seq_tokens_a.append("[CLS]")
    segment_ids_a.append(0)
    for token in tokens_a:
        seq_tokens_a.append(token)
        segment_ids_a.append(0)
    input_ids_a = tokenizer.convert_tokens_to_ids(seq_tokens_a)
    phrase_start_index = len(input_ids_a)
    phrase_end_index = phrase_start_index + len(phrase_nodes)

    for p in phrase_nodes:
        input_ids_a.append(p)
        segment_ids_a.append(phrase_type_id)
        # segment_ids_a.append(0)

    input_ids_a.append(tokenizer.vocab["[SEP]"])
    segment_ids_a.append(0)

    input_ids_a.extend(tokenizer.convert_tokens_to_ids(tokens_ans))
    segment_ids_a.extend([1]*len(tokens_ans))
    input_ids_a.append(tokenizer.vocab["[SEP]"])
    segment_ids_a.append(0)

    seq_tokens_b = []
    segment_ids_b = []
    seq_tokens_b.append("[CLS]")
    segment_ids_b.append(1)
    if tokens_b:
        assert len(tokens_b) > 0
        segment_ids_b.extend([1]*len(tokens_b))
        seq_tokens_b.extend(tokens_b)
    
    input_ids_b = tokenizer.convert_tokens_to_ids(seq_tokens_b)

    input_ids_b.append(tokenizer.vocab["[SEP]"])
    segment_ids_b.append(1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask_a = [1] * len(input_ids_a)
    input_mask_b = [1] * len(input_ids_b)

    # Zero-pad up to the sequence length.
    max_seq_length += num_phrases
    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(0)
        input_mask_a.append(0)
        segment_ids_a.append(0)
        lm_label_ids_a.append(-1)
        phrase_lm_labels.append(-1)

    while len(input_ids_b) < args.max_tag_length:
        input_ids_b.append(0)
        input_mask_b.append(0)
        segment_ids_b.append(1)
        lm_label_ids_b.append(-1)

    # s = "wrong length as maximal {}, input_ids {}, input_mask {}, segment{}, lm_label{}".format(max_seq_length, len(input_ids), len(input_mask), len(segment_ids), len(lm_label_ids))
    s = 'not valid sequence length, please check'
    assert len(input_ids_a) == max_seq_length, s
    assert len(input_mask_a) == max_seq_length, s
    assert len(segment_ids_a) == max_seq_length, s
    assert len(lm_label_ids_a) == max_seq_length, s
    assert len(phrase_lm_labels) == max_seq_length, s

    # image features
    image_start_index = len(input_ids_a) # input_ids_a here for the concated sequence
    image_end_index = image_start_index + img_feat_len
    if args.max_img_seq_length > 0:
        if img_feat_len > args.max_img_seq_length:
            input_mask_b = input_mask_b + [1] * img_feat_len
        else:
            input_mask_b = input_mask_b + [1] * img_feat_len
            pad_img_feat_len = args.max_img_seq_length - img_feat_len
            input_mask_b = input_mask_b + ([0] * pad_img_feat_len)

    lm_label_ids_b = lm_label_ids_b + [-1] * args.max_img_seq_length

    if example.guid < 1:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens_a: %s" % " ".join([str(x) for x in seq_tokens_a]))
        logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
        logging.info("segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))
        logging.info("LM label seq A: %s " % lm_label_ids_a)
        logging.info("phrase masked labels: %s" % " ".join([str(x) for x in phrase_lm_labels]))
        logging.info("Is next sentence label: %s " % example.is_next)
        logging.info("tokens_b: %s" % " ".join([str(x) for x in seq_tokens_b]))
        logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
        logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
        logging.info("segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))
        logging.info("LM label seq B: %s " % lm_label_ids_b)
        # logging.info("Is next sentence label: %s " % example.is_next)

    features = InputFeatures(input_ids_a=input_ids_a,
                             input_mask_a=input_mask_a,
                             segment_ids_a=segment_ids_a,
                             lm_label_ids_a=lm_label_ids_a,
                             is_next=example.is_next,
                             input_ids_b=input_ids_b,
                             input_mask_b=input_mask_b,
                             segment_ids_b=segment_ids_b,
                             lm_label_ids_b=lm_label_ids_b,
                             img_feat_len=img_feat_len,
                             is_img_match=example.is_img_match,
                             phrases_index = [phrase_start_index, phrase_end_index],
                             image_index = [image_start_index, image_end_index],
                             phrase_mask_label=phrase_lm_labels)
    return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def text_concept_extract(text, concept_list):
    """TODO: how to extract concepts from the text, and the candidate list"""
    ## simple version 1, use high_frequence words + POS
    return None


class TextOnlyDataset(Dataset):
    def __init__(self, input_tsv, args, seq_len, tokenizer):
        if input_tsv.endswith('.tsv'):
            logging.info('Loading text only dataset under tsv format')
            self.is_tsv = True
            self.txt_tsv = TSVFile(input_tsv)
        else:
            logging.info('Loading text only dataset under huggingface datasets \
             format under {}'.format(input_tsv))
            self.is_tsv = False
            self.txt_tsv = datasets.load_from_disk(input_tsv, keep_in_memory=False)
            if hasattr(self.txt_tsv, 'keys'):
                # a dataset dict
                self.txt_tsv = self.txt_tsv['train']
        self.sample_count = 0
        self.args = args
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.sample_counter = 0

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if self.is_tsv:
            row = self.txt_tsv.seek(item)
            txt_info = row[0].split('_')
            t1 = row[1]
        else:
            t1 = self.txt_tsv[item]['text']
            if item+1 < self.txt_tsv.num_rows:
                t1 += ' '+self.txt_tsv[item+1]['text']
        # print(item, row)

        t2 = ''
        is_next_label = -1
        is_img_match = -1

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b = self.tokenizer.tokenize(t2)
        else:
            tokens_b = None

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, self.seq_len-3)
        else:
            if len(tokens_a) > self.seq_len - 2:
                tokens_a = tokens_a[:(self.seq_len-2)]

        # transform sample to features
        tokens_a, t1_label = random_word(tokens_a, self.tokenizer)

        if tokens_b:
            if not self.args.mask_loss_for_unmatched and is_next_label == 1:
                t2_label = [-1]*len(tokens_b)
            else:
                tokens_b, t2_label = random_word(tokens_b, self.tokenizer)

        if tokens_b:
            lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]
        else:
            lm_label_ids = [-1] + t1_label + [-1]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0]*len(tokens)

        if tokens_b:
            assert len(tokens_b) > 0
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1]*(len(tokens_b)+1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        # get image feature
        img_feat = torch.zeros(self.args.max_img_seq_length, self.args.img_feature_dim)
        img_feat_len = 0
        lm_label_ids = lm_label_ids + [-1] * self.args.max_img_seq_length
        input_mask += [0] * self.args.max_img_seq_length

        if self.args.visual_learning:
            target_img_feat = img_feat.clone()
            visual_labels = [-1]*self.args.max_img_seq_length
            mask_region_id = [0]*self.args.max_img_seq_length


        if cur_id <= 1:
            logging.info("*** Example ***")
            logging.info("guid: %s" % cur_id)
            logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("LM label: %s " % lm_label_ids)
            logging.info("Is next sentence label: %s " % is_next_label)



        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match),
                item)
        else:
            return img_feat, (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match)
                ), item

    def __len__(self):
        if self.is_tsv:
            return len(self.txt_tsv)
        else:
            return self.txt_tsv.num_rows


class TextOnlyDataset2(Dataset):
    # text only dataset with full length as text
    def __init__(self, input_tsv, args, seq_len, tokenizer):
        print('text only dataset version V2!')
        if input_tsv.endswith('.tsv'):
            logging.info('Loading text only dataset under tsv format')
            self.is_tsv = True
            self.txt_tsv = TSVFile(input_tsv)
        else:
            logging.info('Loading text only dataset under huggingface datasets \
             format under {}'.format(input_tsv))
            self.is_tsv = False
            self.txt_tsv = datasets.load_from_disk(input_tsv)
            if hasattr(self.txt_tsv, 'keys'):
                # a dataset dict
                self.txt_tsv = self.txt_tsv['train']
        self.sample_count = 0
        self.args = args
        self.seq_len = seq_len + args.max_img_seq_length - 1
        self.img_seq_len = 1
        self.tokenizer = tokenizer
        self.sample_counter = 0

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if self.is_tsv:
            row = self.txt_tsv.seek(item)
            txt_info = row[0].split('_')
            t1 = row[1]
        else:
            t1 = self.txt_tsv[item]['text']
            tokens_a = self.tokenizer.tokenize(t1)
            p_id = 1
            while len(tokens_a)<self.seq_len-2 and item+p_id < self.txt_tsv.num_rows:
                # t1 += ' '+self.txt_tsv[item+1]['text']
                tokens_a += self.tokenizer.tokenize(self.txt_tsv[item+p_id]['text'])
                p_id += 1
                if p_id > 10:
                    break
                    print('looping for more than {} times now!'.format(p_id))
        # print(item, row)

        t2 = ''
        is_next_label = -1
        is_img_match = -1

        # tokenize
        # tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b = self.tokenizer.tokenize(t2)
        else:
            tokens_b = None

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, self.seq_len-3)
        else:
            if len(tokens_a) > self.seq_len - 2:
                tokens_a = tokens_a[:(self.seq_len-2)]

        # transform sample to features
        tokens_a, t1_label = random_word(tokens_a, self.tokenizer)

        if tokens_b:
            if not self.args.mask_loss_for_unmatched and is_next_label == 1:
                t2_label = [-1]*len(tokens_b)
            else:
                tokens_b, t2_label = random_word(tokens_b, self.tokenizer)

        if tokens_b:
            lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]
        else:
            lm_label_ids = [-1] + t1_label + [-1]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0]*len(tokens)

        if tokens_b:
            assert len(tokens_b) > 0
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1]*(len(tokens_b)+1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        # get image feature
        if self.img_seq_len > 0:
            img_feat = torch.zeros(self.img_seq_len, self.args.img_feature_dim)
            img_feat_len = 0
            lm_label_ids = lm_label_ids + [-1] * self.img_seq_len
            input_mask += [0] * self.img_seq_len

            if self.args.visual_learning:
                target_img_feat = img_feat.clone()
                visual_labels = [-1]*self.img_seq_len
                mask_region_id = [0]*self.img_seq_len
        else:
            img_feat = None
            target_img_feat = None
            visual_labels = None
            mask_region_id = None

        if cur_id <= 1:
            logging.info("*** Example ***")
            logging.info("guid: %s" % cur_id)
            logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("LM label: %s " % lm_label_ids)
            logging.info("Is next sentence label: %s " % is_next_label)

        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match),
                item)
        else:
            return img_feat, (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match)
                ), item

    def __len__(self):
        if self.is_tsv:
            return len(self.txt_tsv)
        else:
            return self.txt_tsv.num_rows
        