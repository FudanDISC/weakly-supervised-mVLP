'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from locale import normalize
from turtle import pos
from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertForMaskedLM, BertOnlyMLMHead
from transformers import XLMWithLMHeadModel, XLMRobertaForMaskedLM
from .model_utils import get_pos_neg_sims, get_sims_from_mats_s2t, get_sims_from_mats_t2s

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
from oscar.utils.misc import mkdir, get_rank

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                       

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = sim_i2t.t()

        # sim_targets = torch.zeros(sim_i2t.size()).to(sim_i2t.device).to(sim_i2t.dtype)
        sim_targets = torch.zeros_like(sim_i2t)
        sim_targets.fill_diagonal_(1)
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        # image_embeds_neg = []    
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     image_embeds_neg.append(image_embeds[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg,dim=0)
        hard_img_index = torch.multinomial(weights_t2i, 1).squeeze()
        image_embeds_neg = torch.index_select(image_embeds, dim=0, index=hard_img_index)

        # select a negative text for each image
        # text_embeds_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_embeds_neg.append(text_embeds[neg_idx])
        #     text_atts_neg.append(text.attention_mask[neg_idx])
        # text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        # text_atts_neg = torch.stack(text_atts_neg,dim=0)      
        hard_text_index = torch.multinomial(weights_i2t, 1).squeeze()
        text_embeds_neg = torch.index_select(text_embeds, dim=0, index=hard_text_index)
        text_atts_neg = torch.index_select(text.attention_mask, dim=0, index=hard_text_index)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
        
        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
         
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = None,
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss        

        return loss_mlm, loss_ita, loss_itm  

        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


class ALBEF_fast(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
        self.avoid_nan = config['avoid_nan']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config, ignore_mismatched_sizes=True)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)

    def model_part(self, part_name):
        tmp_config = self.text_encoder.bert.config
        num_layers = tmp_config.num_hidden_layers
        fusion_layers = tmp_config.fusion_layer
        if part_name == 'txt_encoder':
            return [self.text_encoder.bert.encoder.layer[i] for i in range(0, fusion_layers)] + [self.text_proj]
        elif part_name == 'vis_encoder':
            return [self.visual_encoder, self.vision_proj]
        elif part_name == 'embedding':
            return [self.text_encoder.bert.embeddings]
        elif part_name == 'fusion_encoder':
            return [self.text_encoder.bert.encoder.layer[i] for i in range(fusion_layers, num_layers)]
        elif part_name == 'task_head':
            return [self.text_encoder.cls, self.itm_head]
        else:
            raise ValueError

    def freeze(self, freeze_part):
        modules = self.model_part(freeze_part)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze(self, freeze_part):
        modules = self.model_part(freeze_part)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True       

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                       

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = sim_i2t.t()

        # sim_targets = torch.zeros(sim_i2t.size()).to(sim_i2t.device).to(sim_i2t.dtype)
        sim_targets = torch.zeros_like(sim_i2t)
        sim_targets.fill_diagonal_(1)
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        # output_hidden_states=True,
                                        # output_attentions=True,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        # image_embeds_neg = []    
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     image_embeds_neg.append(image_embeds[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg,dim=0)
        try:
            hard_img_index = torch.multinomial(weights_t2i, 1).squeeze()
        except:
            weights_t2i.fill_diagonal_(-1)
            hard_img_index = torch.max(weights_t2i, dim=1)[1]
        image_embeds_neg = torch.index_select(image_embeds, dim=0, index=hard_img_index)

        # select a negative text for each image
        # text_embeds_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_embeds_neg.append(text_embeds[neg_idx])
        #     text_atts_neg.append(text.attention_mask[neg_idx])
        # text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        # text_atts_neg = torch.stack(text_atts_neg,dim=0)
        try:      
            hard_text_index = torch.multinomial(weights_i2t, 1).squeeze()
        except:
            weights_i2t.fill_diagonal_(-1)
            hard_text_index = torch.max(weights_i2t, dim=1)[1]
        text_embeds_neg = torch.index_select(text_embeds, dim=0, index=hard_text_index)
        text_atts_neg = torch.index_select(text.attention_mask, dim=0, index=hard_text_index)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        # output_hidden_states=True,
                                        # output_attentions=True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        if self.avoid_nan:
            loss_itm = F.cross_entropy(vl_output, itm_labels, reduction='none')
            loss_itm = torch.mean(loss_itm[~(torch.isnan(loss_itm))])
        else:
            loss_itm = F.cross_entropy(vl_output, itm_labels)   
        
        ##================= MLM ========================##                
        # input_ids = text.input_ids.clone()
        # labels = input_ids.clone()

        # probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        # input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
        #                               probability_matrix = probability_matrix) 
         
        mlm_output = self.text_encoder(encoder_embeds=text_embeds, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = labels,   
                                       soft_labels = None,
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss

        if get_rank()==0 and (torch.isnan(loss_mlm) or torch.isnan(loss_itm) or torch.isnan(loss_ita)):
            random_name = random.random()
            debug_info = {'input': {'input_ids': text.input_ids, 'images': image, 'mased_ids':input_ids, 'txt_attn_mask':text.attention_mask}}
            debug_info['loss'] = {'itm': loss_itm, 'mlm': loss_mlm, 'ita': loss_ita}
            debug_info['single-modal'] = {'image': image_embeds, 'text': text_embeds, 'ita_sim': sim_i2t, 'hard_img_index':hard_img_index, 'hard_txt_index':hard_text_index}
            debug_info['itm_out'] = {'itm_logit': vl_output, 'labels': itm_labels, 'last hidden states':torch.cat([output_pos.last_hidden_state, output_neg.last_hidden_state],dim=0), \
                'pos hidden':output_pos.hidden_states, 'neg hidden': output_neg.hidden_states}
            debug_info['mlm_out'] = {'mlm_logit': mlm_output.logits, 'labels': labels, 'hidden states': mlm_output.hidden_states}
            torch.save(debug_info, '/remote-home/zjli/tmp_debug/debug_info_{:.4f}.pth'.format(random_name))
            raise ValueError

        return loss_mlm, loss_ita, loss_itm  

        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

        
class ALBEF_Stage1(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        if 'conditional_mlm_probability' in config:
            self.cond_mlm_probability = config['conditional_mlm_probability']
        else:
            self.cond_mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
        self.avoid_nan = config['avoid_nan']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        print(bert_config)
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config, ignore_mismatched_sizes=True)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)

        # use MLM on top of the text encoder
        use_half_mlm = False
        txt_teacher = False
        if 'text_modeling' in config:
            if config['text_modeling']['enable']:
                print('using text modeling')
            txt_model_config = config['text_modeling']
            use_half_mlm = txt_model_config['enable']
            txt_teacher = txt_model_config['teacher']
            if txt_teacher:
                self.txt_teacher_class = txt_model_config['teacher_class']
                if self.txt_teacher_class == 'xlm':
                    model_class = XLMWithLMHeadModel
                elif self.txt_teacher_class == 'xlm-r':
                    model_class = XLMRobertaForMaskedLM
                else:
                    raise NotImplementedError
                print('learning from teacher: {}'.format(txt_model_config['teacher_name']))
                self.txt_teacher_model = model_class.from_pretrained(txt_model_config['teacher_name'])
                # print(self.txt_teacher_model.transformer.position_embeddings.weight)
                self.teacher_alpha = txt_model_config['teacher_alpha']
            else:
                self.teacher_alpha = 0
                self.txt_teacher_model = None       
        else:
            self.teacher_alpha = 0
            txt_model_config = None
            self.txt_teacher_model = None
        
        if use_half_mlm:
            self.half_mlm_head = BertOnlyMLMHead(bert_config)
            self.text_modeling = True
        else:
            self.half_mlm_head = None
            self.text_modeling = False

    def model_part(self, part_name):
        tmp_config = self.text_encoder.bert.config
        num_layers = tmp_config.num_hidden_layers
        fusion_layers = tmp_config.fusion_layer
        if part_name == 'txt_encoder':
            return [self.text_encoder.bert.encoder.layer[i] for i in range(0, fusion_layers)] + [self.text_proj]
        elif part_name == 'vis_encoder':
            return [self.visual_encoder, self.vision_proj]
        elif part_name == 'embedding':
            return [self.text_encoder.bert.embeddings]
        elif part_name == 'fusion_encoder':
            return [self.text_encoder.bert.encoder.layer[i] for i in range(fusion_layers, num_layers)]
        elif part_name == 'task_head':
            return [self.text_encoder.cls, self.itm_head]
        else:
            raise ValueError

    def freeze(self, freeze_part):
        modules = self.model_part(freeze_part)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze(self, freeze_part):
        modules = self.model_part(freeze_part)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True  

    def forward(self, image=None, text=None, text_trans=None, text_langs=None, text_trans_langs=None, mode='img-txt-contras', token_sim_mode='max', token_sim_k=3):
        # assert mode in ['img-txt', 'txt-txt']
        if mode == 'img-txt-contras':
            return self.forward_img_txt_contras(image, text)
        elif mode == 'txt-txt-contras':
            return self.forward_txt_trans(text, text_trans)
        elif mode == 'txt-txt-full':
            return self.forward_txt_full(text, text_trans, text_langs=text_langs, 
                                    text_trans_lang=text_trans_langs)
        elif mode == 'img-txt-full':
            return self.forward_img_txt_full(image, text)
        elif mode == 'txt-txt-tlm':
            return self.forward_txt_tlm(text, text_trans=text_trans, text_langs=text_langs, text_trans_langs=text_trans_langs)
        elif mode == 'txt-txt-wla':
            return self.forward_txt_trans_weak_align(text, text_trans=text_trans, token_sim_method=token_sim_mode, sim_k=token_sim_k)
        elif mode == 'para_txt_full':
            return self.forward_para_txt_full(text, text_trans)
        elif mode == 'para_txt_abl-xcl':
            return self.forward_para_txt_abl(text, text_trans)
        elif mode == 'mono_txt':
            return self.forward_mono_txt(text)
        else:
            raise NotImplementedError

    def forward_txt_trans(self, text, text_trans):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        trans_text_output = self.text_encoder.bert(text_trans.input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        trans_text_embeds = trans_text_output.last_hidden_state
        trans_text_feat = F.normalize(self.text_proj(trans_text_embeds[:,0,:]),dim=-1)

        # s for source, t for target
        sim_s2t = text_feat @ trans_text_feat.t() / self.temp
        sim_t2s = sim_s2t.t()

        sim_targets = torch.zeros_like(sim_s2t)
        sim_targets.fill_diagonal_(1)

        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_s2t+loss_t2s)/2

        return (loss_ita,)

    def forward_txt_trans_weak_align(self, text, text_trans, token_sim_method='max', sim_k=3):
        # token-level weakly-supervised alignment
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_embeds = F.normalize(self.text_proj(text_embeds),dim=-1)
        text_feat = text_embeds[:,0,:]

        trans_text_output = self.text_encoder.bert(text_trans.input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        trans_text_embeds = trans_text_output.last_hidden_state
        trans_text_embeds = F.normalize(self.text_proj(trans_text_embeds),dim=-1)
        trans_text_feat = trans_text_embeds[:,0,:]

        # s for source, t for target
        sim_s2t = text_feat @ trans_text_feat.t() / self.temp
        sim_t2s = sim_s2t.t()

        sim_targets = torch.zeros_like(sim_s2t)
        sim_targets.fill_diagonal_(1)

        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_s2t+loss_t2s)/2

        with torch.no_grad():     
            weights_s2t = F.softmax(sim_s2t,dim=1)
            weights_t2s = F.softmax(sim_t2s,dim=1)
   
            weights_s2t.fill_diagonal_(0)
            weights_t2s.fill_diagonal_(0)

        try:
            hard_src_index = torch.multinomial(weights_t2s, 1).squeeze()
        except:
            weights_t2s.fill_diagonal_(-1)
            hard_src_index = torch.max(weights_t2s, dim=1)[1]
        hard_src_emb = torch.index_select(text_embeds, dim=0, index=hard_src_index)
        hard_src_attn = torch.index_select(text.attention_mask, dim=0, index=hard_src_index)

        try:      
            hard_tar_index = torch.multinomial(weights_s2t, 1).squeeze()
        except:
            weights_s2t.fill_diagonal_(-1)
            hard_tar_index = torch.max(weights_s2t, dim=1)[1]
        hard_tar_emb = torch.index_select(trans_text_embeds, dim=0, index=hard_tar_index)
        hard_tar_attn = torch.index_select(text_trans.attention_mask, dim=0, index=hard_tar_index)

        pos_sim_mats = torch.bmm(text_embeds, trans_text_embeds.permute(0, 2, 1)) # batch src to target
        neg_s2t_sim_mats = torch.bmm(text_embeds, hard_tar_emb.permute(0, 2, 1))
        neg_t2s_sim_mats = torch.bmm(hard_src_emb, trans_text_embeds.permute(0, 2, 1))

        # quick version for wla
        pos_weak_sims_s2t, pos_weak_sims_t2s = get_sims_from_mats_s2t(pos_sim_mats, text.attention_mask, text_trans.attention_mask, bi_direction=True, sim_method=token_sim_method, k=sim_k)
        neg_weak_sims_s2t = get_sims_from_mats_s2t(neg_s2t_sim_mats, text.attention_mask, hard_tar_attn, bi_direction=False, sim_method=token_sim_method, k=sim_k)
        neg_weak_sims_t2s = get_sims_from_mats_t2s(neg_t2s_sim_mats, hard_src_attn, text_trans.attention_mask, sim_method=token_sim_method, k=sim_k)

        # weakly-aligned token-level loss (slow version below)
        # last_hidden_dim = text_embeds.shape[-1]
        # valid_src_tokens = F.normalize(torch.masked_select(text_embeds, text.attention_mask.unsqueeze(-1)>0).reshape(-1, last_hidden_dim), dim=-1)
        # valid_tar_tokens = F.normalize(torch.masked_select(trans_text_embeds, text_trans.attention_mask.unsqueeze(-1)>0).reshape(-1, last_hidden_dim), dim=-1)
        # token_sims = valid_src_tokens @ valid_tar_tokens.t()
        # pos_weak_sims_s2t, neg_weak_sims_s2t = get_pos_neg_sims(token_sims, src_mask=text.attention_mask, tar_mask=text_trans.attention_mask, sim_method=token_sim_method, k=sim_k)
        # pos_weak_sims_t2s, neg_weak_sims_t2s = get_pos_neg_sims(token_sims.t(), src_mask=text_trans.attention_mask, tar_mask=text.attention_mask, sim_method=token_sim_method, k=sim_k)
        
        # loss computation
        wta_loss_s2t = torch.mean(torch.clamp(neg_weak_sims_s2t + 0.2 - pos_weak_sims_s2t, min=0))
        wta_loss_t2s = torch.mean(torch.clamp(neg_weak_sims_t2s + 0.2 - pos_weak_sims_t2s, min=0))
        wta_loss = (wta_loss_s2t + wta_loss_t2s) / 2

        return (loss_ita, wta_loss)

    def forward_txt_full(self, text, text_trans, text_langs=None, text_trans_lang=None):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        trans_input_ids = text_trans.input_ids.clone()
        trans_labels = trans_input_ids.clone()

        probability_matrix = torch.full(trans_labels.shape, self.mlm_probability)    

        trans_input_ids, trans_labels = self.mask(trans_input_ids, self.text_encoder.config.vocab_size, trans_input_ids.device, targets=trans_labels,
                                      probability_matrix = probability_matrix) 

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        trans_text_output = self.text_encoder.bert(trans_input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        trans_text_embeds = trans_text_output.last_hidden_state
        trans_text_feat = F.normalize(self.text_proj(trans_text_embeds[:,0,:]),dim=-1)

        # s for source, t for target
        sim_s2t = text_feat @ trans_text_feat.t() / self.temp
        sim_t2s = sim_s2t.t()

        sim_targets = torch.zeros_like(sim_s2t)
        sim_targets.fill_diagonal_(1)

        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_s2t+loss_t2s)/2

        assert self.half_mlm_head is not None
        lm_mask = trans_labels >= 0
        hidden_size = trans_text_embeds.shape[-1]
        masked_sequence_output = torch.masked_select(trans_text_embeds, lm_mask.unsqueeze(-1)).reshape(-1, hidden_size)
        masked_labels = torch.masked_select(trans_labels, lm_mask).reshape(-1)
        half_prediction_scores = self.half_mlm_head(masked_sequence_output)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(half_prediction_scores, masked_labels)

        if self.txt_teacher_model is not None:
            with torch.no_grad():
                if self.txt_teacher_class == 'xlm':
                    teacher_outputs = self.txt_teacher_model(trans_input_ids, attention_mask = text_trans.attention_mask,                      
                                        langs=text_trans_lang, return_dict = True)
                elif self.txt_teacher_class == 'xlm-r':
                    teacher_outputs = self.txt_teacher_model(trans_input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True)
                else:
                    raise NotImplementedError
                teacher_logits = teacher_outputs.logits
                # print(teacher_outputs.loss)
                teacher_vocab_size = teacher_logits.shape[-1]
                assert half_prediction_scores.shape[-1] == teacher_vocab_size
                teacher_logits = torch.masked_select(teacher_logits, lm_mask.unsqueeze(-1)).reshape(-1, teacher_vocab_size)
                teacher_labels = F.softmax(teacher_logits, dim=-1)
        else:
            teacher_labels = None

        if teacher_labels is not None:
            loss_distill = -torch.sum(F.log_softmax(half_prediction_scores, dim=-1)*teacher_labels,dim=-1)
            loss_distill = loss_distill.mean()
            masked_lm_loss = (1-self.teacher_alpha)*masked_lm_loss + self.teacher_alpha*loss_distill

        return (loss_ita, masked_lm_loss) 

    def forward_txt_tlm(self, text, text_trans, text_langs=None, text_trans_langs=None):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        trans_text_output = self.text_encoder.bert(text_trans.input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        trans_text_embeds = trans_text_output.last_hidden_state
        trans_text_feat = F.normalize(self.text_proj(trans_text_embeds[:,0,:]),dim=-1)

        # s for source, t for target
        sim_s2t = text_feat @ trans_text_feat.t() / self.temp
        sim_t2s = sim_s2t.t()

        sim_targets = torch.zeros_like(sim_s2t)
        sim_targets.fill_diagonal_(1)

        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_s2t+loss_t2s)/2

        # TLM Modeling

        # perform concate pair
        
        input_ids, attn_mask, pos_ids, lang_ids = self.concate_pair(text, text_trans, text_lang=text_langs, trans_lang=text_trans_langs)
        # if get_rank()==0:
        #     print('text')
        #     print(text)
        #     print('translation text')
        #     print(text_trans)
        #     print('after concat')
        #     print(input_ids)
        #     print(attn_mask)
        #     print(pos_ids)
        #     print(lang_ids)
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)    
        
        # masking
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, input_ids.device, targets=labels,
                                      probability_matrix = probability_matrix)

        # textual model forwarding
        full_text_output = self.text_encoder.bert(input_ids, attention_mask = attn_mask,                      
                                        return_dict = True, mode = 'text')            
        full_text_embeds = full_text_output.last_hidden_state
        assert self.half_mlm_head is not None

        lm_mask = labels >= 0
        hidden_size = full_text_embeds.shape[-1]
        # print('child size', full_text_embeds.shape)
        masked_sequence_output = torch.masked_select(full_text_embeds, lm_mask.unsqueeze(-1)).reshape(-1, hidden_size)
        masked_labels = torch.masked_select(labels, lm_mask).reshape(-1)
        half_prediction_scores = self.half_mlm_head(masked_sequence_output)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(half_prediction_scores, masked_labels)

        if self.txt_teacher_model is not None:
            with torch.no_grad():
                if self.txt_teacher_class == 'xlm':
                    # only support xlm for TLM
                    teacher_outputs = self.txt_teacher_model(input_ids, attention_mask = attn_mask, position_ids=pos_ids,                  
                                        langs=lang_ids, labels=None, return_dict = True, quick_mask=lm_mask)
                # elif self.txt_teacher_class == 'xlm-r':
                #     teacher_outputs = self.txt_teacher_model()
                else:
                    raise NotImplementedError
                teacher_logits = teacher_outputs.logits
                # print(teacher_outputs.loss)
                # teacher_labels = F.softmax(teacher_logits, dim=-1)
                # teacher_logits = teacher_outputs.logits
                teacher_vocab_size = teacher_logits.shape[-1]
                assert half_prediction_scores.shape[-1] == teacher_vocab_size
                # print('teacher size', teacher_logits.shape)
                # teacher_logits = torch.masked_select(teacher_logits, lm_mask.unsqueeze(-1)).reshape(-1, teacher_vocab_size)
                teacher_labels = F.softmax(teacher_logits, dim=-1)
                # print(teacher_labels[torch.arange(teacher_labels.shape[0]), masked_labels])
        else:
            teacher_labels = None
        
        # loss_distill = -torch.sum(F.log_softmax(half_prediction_scores, dim=-1)*teacher_labels,dim=-1)
        # loss_distill = loss_distill[labels!=-100].mean()
        # masked_lm_loss = (1-self.teacher_alpha)*masked_lm_loss + self.teacher_alpha*loss_distill

        if teacher_labels is not None:
            loss_distill = -torch.sum(F.log_softmax(half_prediction_scores, dim=-1)*teacher_labels,dim=-1)
            loss_distill = loss_distill.mean()
            masked_lm_loss = (1-self.teacher_alpha)*masked_lm_loss + self.teacher_alpha*loss_distill

        return (loss_ita, masked_lm_loss)

    def forward_para_txt_full(self, text, text_trans):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        # for source text, mask and perform single modality embedding
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.cond_mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, labels.device, targets=labels,
                                      probability_matrix = probability_matrix)

        text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        # for target text, mask and perform single modality embedding
        trans_input_ids = text_trans.input_ids.clone()
        trans_labels = trans_input_ids.clone()

        trans_probability_matrix = torch.full(trans_labels.shape, self.cond_mlm_probability) 
        trans_input_ids, trans_labels = self.mask(trans_input_ids, self.text_encoder.config.vocab_size, trans_labels.device, targets=trans_labels,
                                      probability_matrix = trans_probability_matrix)
        trans_text_output = self.text_encoder.bert(trans_input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        trans_text_embeds = trans_text_output.last_hidden_state
        trans_text_feat = F.normalize(self.text_proj(trans_text_embeds[:,0,:]),dim=-1)

        # s for source, t for target
        sim_s2t = text_feat @ trans_text_feat.t() / self.temp
        sim_t2s = sim_s2t.t()

        sim_targets = torch.zeros_like(sim_s2t)
        sim_targets.fill_diagonal_(1)

        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_s2t+loss_t2s)/2

        s2t_mlm_output = self.text_encoder(encoder_embeds=text_embeds, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = trans_text_embeds,
                                       encoder_attention_mask = text_trans.attention_mask,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = labels,   
                                       soft_labels = None,
                                      )
        
        t2s_mlm_output = self.text_encoder(encoder_embeds=trans_text_embeds, 
                                       attention_mask = text_trans.attention_mask,
                                       encoder_hidden_states = text_embeds,
                                       encoder_attention_mask = text.attention_mask,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = trans_labels,   
                                       soft_labels = None,
                                      )   

        return (loss_ita, s2t_mlm_output.loss, t2s_mlm_output.loss)

    def forward_para_txt_abl(self, text, text_trans):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        # for source text, mask and perform single modality embedding
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.cond_mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, labels.device, targets=labels,
                                      probability_matrix = probability_matrix)

        text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        # text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        # for target text, mask and perform single modality embedding
        trans_input_ids = text_trans.input_ids.clone()
        trans_labels = trans_input_ids.clone()

        trans_probability_matrix = torch.full(trans_labels.shape, self.cond_mlm_probability) 
        trans_input_ids, trans_labels = self.mask(trans_input_ids, self.text_encoder.config.vocab_size, trans_labels.device, targets=trans_labels,
                                      probability_matrix = trans_probability_matrix)
        trans_text_output = self.text_encoder.bert(trans_input_ids, attention_mask = text_trans.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        trans_text_embeds = trans_text_output.last_hidden_state
        # trans_text_feat = F.normalize(self.text_proj(trans_text_embeds[:,0,:]),dim=-1)

        # s for source, t for target
        # sim_s2t = text_feat @ trans_text_feat.t() / self.temp
        # sim_t2s = sim_s2t.t()

        # sim_targets = torch.zeros_like(sim_s2t)
        # sim_targets.fill_diagonal_(1)

        # loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_targets,dim=1).mean()
        # loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_targets,dim=1).mean() 

        # loss_ita = (loss_s2t+loss_t2s)/2

        s2t_mlm_output = self.text_encoder(encoder_embeds=text_embeds, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = trans_text_embeds,
                                       encoder_attention_mask = text_trans.attention_mask,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = labels,   
                                       soft_labels = None,
                                      )
        
        t2s_mlm_output = self.text_encoder(encoder_embeds=trans_text_embeds, 
                                       attention_mask = text_trans.attention_mask,
                                       encoder_hidden_states = text_embeds,
                                       encoder_attention_mask = text.attention_mask,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = trans_labels,   
                                       soft_labels = None,
                                      )   

        return (s2t_mlm_output.loss, t2s_mlm_output.loss)
    
    def forward_mono_txt(self, text):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, labels.device, targets=labels,
                                      probability_matrix = probability_matrix)

        text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        mlm_output = self.text_encoder(encoder_embeds=text_embeds, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = None,
                                       encoder_attention_mask = None,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = labels,   
                                       soft_labels = None,
                                       skip_cross_attention=True,
                                      )
        return (mlm_output.loss,)

    def forward_mono_txt_debug(self, text):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, labels.device, targets=labels,
                                      probability_matrix = probability_matrix)

        text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True)            
        text_embeds = text_output.last_hidden_state
        mlm_output = self.text_encoder(encoder_embeds=text_embeds, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = None,
                                       encoder_attention_mask = None,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = labels,   
                                       soft_labels = None,
                                       output_hidden_states=True,
                                       skip_cross_attention=True,
                                      )
        return ((text_output,mlm_output), (input_ids, labels))

    
    def forward_img_txt_contras(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                       

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = sim_i2t.t()

        # sim_targets = torch.zeros(sim_i2t.size()).to(sim_i2t.device).to(sim_i2t.dtype)
        sim_targets = torch.zeros_like(sim_i2t)
        sim_targets.fill_diagonal_(1)
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        return (loss_ita,)

    

    def forward_img_txt_full(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.cond_mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                       

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = sim_i2t.t()

        # sim_targets = torch.zeros(sim_i2t.size()).to(sim_i2t.device).to(sim_i2t.dtype)
        sim_targets = torch.zeros_like(sim_i2t)
        sim_targets.fill_diagonal_(1)
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        # output_hidden_states=True,
                                        # output_attentions=True,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        # image_embeds_neg = []    
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     image_embeds_neg.append(image_embeds[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg,dim=0)
        try:
            hard_img_index = torch.multinomial(weights_t2i, 1).squeeze()
        except:
            weights_t2i.fill_diagonal_(-1)
            hard_img_index = torch.max(weights_t2i, dim=1)[1]
        image_embeds_neg = torch.index_select(image_embeds, dim=0, index=hard_img_index)

        # select a negative text for each image
        # text_embeds_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_embeds_neg.append(text_embeds[neg_idx])
        #     text_atts_neg.append(text.attention_mask[neg_idx])
        # text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        # text_atts_neg = torch.stack(text_atts_neg,dim=0)
        try:      
            hard_text_index = torch.multinomial(weights_i2t, 1).squeeze()
        except:
            weights_i2t.fill_diagonal_(-1)
            hard_text_index = torch.max(weights_i2t, dim=1)[1]
        text_embeds_neg = torch.index_select(text_embeds, dim=0, index=hard_text_index)
        text_atts_neg = torch.index_select(text.attention_mask, dim=0, index=hard_text_index)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        # output_hidden_states=True,
                                        # output_attentions=True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        if self.avoid_nan:
            loss_itm = F.cross_entropy(vl_output, itm_labels, reduction='none')
            loss_itm = torch.mean(loss_itm[~(torch.isnan(loss_itm))])
        else:
            loss_itm = F.cross_entropy(vl_output, itm_labels)   
        
        ##================= MLM ========================##                
        # input_ids = text.input_ids.clone()
        # labels = input_ids.clone()

        # probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        # input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
        #                               probability_matrix = probability_matrix) 
         
        mlm_output = self.text_encoder(encoder_embeds=text_embeds, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       mode = 'fusion',
                                       labels = labels,   
                                       soft_labels = None,
                                       alpha = alpha,
                                      )                           
        loss_mlm = mlm_output.loss

        if get_rank()==0 and (torch.isnan(loss_mlm) or torch.isnan(loss_itm) or torch.isnan(loss_ita)):
            random_name = random.random()
            debug_info = {'input': {'input_ids': text.input_ids, 'images': image, 'mased_ids':input_ids, 'txt_attn_mask':text.attention_mask}}
            debug_info['loss'] = {'itm': loss_itm, 'mlm': loss_mlm, 'ita': loss_ita}
            debug_info['single-modal'] = {'image': image_embeds, 'text': text_embeds, 'ita_sim': sim_i2t, 'hard_img_index':hard_img_index, 'hard_txt_index':hard_text_index}
            debug_info['itm_out'] = {'itm_logit': vl_output, 'labels': itm_labels, 'last hidden states':torch.cat([output_pos.last_hidden_state, output_neg.last_hidden_state],dim=0), \
                'pos hidden':output_pos.hidden_states, 'neg hidden': output_neg.hidden_states}
            debug_info['mlm_out'] = {'mlm_logit': mlm_output.logits, 'labels': labels, 'hidden states': mlm_output.hidden_states}
            torch.save(debug_info, '/opt/tiger/tmp_debug/debug_info_{:.4f}.pth'.format(random_name))
            raise ValueError

        return loss_mlm, loss_ita, loss_itm  

        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices[input_ids == self.tokenizer.bos_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
    
    def concate_pair(self, text, text_trans, text_lang, trans_lang):
        # concate 2 parts of the input for Translation Language Modeling (available in XLM)
        bs = text.input_ids.shape[0]
        max_len = text.input_ids.shape[1] + text_trans.input_ids.shape[1]
        concat_ids = []
        concat_attn = []
        concat_pos_ids = []
        concat_langs = []
        c_device = text.input_ids.device
        pad_id = self.tokenizer.pad_token_id
        # consider 1-dim language id input
        text_lang = text_lang.expand_as(text.input_ids)
        trans_lang = trans_lang.expand_as(text_trans.input_ids)
        for i in range(bs):
            valid_ids_a = text.attention_mask[i] > 0
            valid_ids_b = text_trans.attention_mask[i] > 0
            input_ids_a = text.input_ids[i][valid_ids_a]
            input_ids_b = text_trans.input_ids[i][valid_ids_b]
            size_a = input_ids_a.shape[0]
            size_b = input_ids_b.shape[0]
            pad_size = max_len - size_a - size_b
            pos_ids_a = torch.arange(size_a, device=c_device)
            pos_ids_b = torch.arange(size_b, device=c_device)
            lang_a = text_lang[i][valid_ids_a]
            lang_b = trans_lang[i][valid_ids_b]
            concat_ids.append(torch.cat([input_ids_a, input_ids_b, torch.ones(pad_size, dtype=torch.long, device=c_device)*pad_id]))
            concat_attn.append(torch.cat([torch.ones(size_a+size_b, dtype=torch.long, device=c_device), torch.zeros(pad_size, dtype=torch.long, device=c_device)]))
            concat_pos_ids.append(torch.cat([pos_ids_a, pos_ids_b, torch.zeros(pad_size, dtype=torch.long, device=c_device)]))
            concat_langs.append(torch.cat([lang_a, lang_b, torch.zeros(pad_size, dtype=torch.long, device=c_device)]))
        return torch.stack(concat_ids), torch.stack(concat_attn), torch.stack(concat_pos_ids), torch.stack(concat_langs)




@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
