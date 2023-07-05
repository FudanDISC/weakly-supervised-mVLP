'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial

from cv2 import log
from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertForMaskedLM, BertModel, BertQAPredictionHead, BertLMHeadModel

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF_CLS(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))            
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.qa_head = BertQAPredictionHead(bert_config, config['answer_num'])

    def model_part(self, part_name):
        tmp_config = self.text_encoder.config
        num_layers = tmp_config.num_hidden_layers
        fusion_layers = tmp_config.fusion_layer
        if part_name == 'txt_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(0, fusion_layers)] + [self.text_proj]
        elif part_name == 'vis_encoder':
            return [self.visual_encoder, self.vision_proj]
        elif part_name == 'embedding':
            return [self.text_encoder.embeddings]
        elif part_name == 'fusion_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(fusion_layers, num_layers)]
        elif part_name == 'task_head':
            return [self.qa_head]
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
    
    def forward(self, image, questions_ids, questions_attn_mask, answers=None, alpha=0):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
 

        text_output = self.text_encoder(questions_ids, attention_mask = questions_attn_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state

        # print(text_embeds.shape, image_embeds.shape, questions.attention_mask.shape, image_atts.shape)

        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = questions_attn_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                                

        vl_embeddings = output_pos.last_hidden_state[:,0,:]
        logits = self.qa_head(vl_embeddings)

        if answers is not None:
            loss_fcn = nn.CrossEntropyLoss()
            # print(logits.device, answers.device)
            loss = loss_fcn(logits, answers.view(-1))
            if torch.isnan(loss):
                if image.device == torch.device('cuda:0'):
                    torch.save({'label':answers, 'logits':logits, 'vl_embeddings': vl_embeddings}, '/remote-home/zjli/tmp_debug/vqa_debug.pt')
                    raise ValueError
            return loss, logits               

        return logits


class ALBEF_GEN(nn.Module):
    # generation based QA model
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
            
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)    

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))             
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)   
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params() 
            self.momentum = 0.995
        

    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [quesiton.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            if self.distill:                    
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    question_output_m = self.text_encoder_m(quesiton.input_ids, 
                                                            attention_mask = quesiton.attention_mask, 
                                                            encoder_hidden_states = image_embeds_m,
                                                            encoder_attention_mask = image_atts,                             
                                                            return_dict = True)    

                    question_states_m = []                
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m.last_hidden_state[b]]*n
                    question_states_m = torch.stack(question_states_m,0)    

                    logits_m = self.text_decoder_m(answer.input_ids, 
                                                   attention_mask = answer.attention_mask, 
                                                   encoder_hidden_states = question_states_m,
                                                   encoder_attention_mask = question_atts,                                  
                                                   return_logits = True,
                                                  )                       

                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  soft_labels = F.softmax(logits_m,dim=-1),
                                                  alpha = alpha,
                                                  reduction = 'none',
                                                 )   
            else:
                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )                      
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            return loss
            

        else: 
            question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)                    
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask, 
                                                    answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
        

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


class ALBEFforClassification(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))            
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        cls_config = config['cls_head']
        if cls_config['head_type'] == 'linear':
            self.cls_head = nn.Linear(text_width, cls_config['class_num'])
        elif cls_config['head_type'] == 'mlp':
            self.cls_head = nn.Sequential(
                    nn.Linear(text_width, text_width * cls_config['cls_hidden_scale']),
                    nn.ReLU(),
                    nn.Linear(text_width * cls_config['cls_hidden_scale'], cls_config['class_num'])
                )

    def model_part(self, part_name):
        tmp_config = self.text_encoder.config
        num_layers = tmp_config.num_hidden_layers
        fusion_layers = tmp_config.fusion_layer
        if part_name == 'txt_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(0, fusion_layers)] + [self.text_proj]
        elif part_name == 'vis_encoder':
            return [self.visual_encoder, self.vision_proj]
        elif part_name == 'embedding':
            return [self.text_encoder.embeddings]
        elif part_name == 'fusion_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(fusion_layers, num_layers)]
        elif part_name == 'task_head':
            return [self.cls_head]
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
    
    def forward(self, image, sent_ids, sent_attn_mask, labels=None, alpha=0):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
 

        text_output = self.text_encoder(sent_ids, attention_mask = sent_attn_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state

        # print(text_embeds.shape, image_embeds.shape, questions.attention_mask.shape, image_atts.shape)

        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = sent_attn_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                                

        vl_embeddings = output_pos.last_hidden_state[:,0,:]
        logits = self.cls_head(vl_embeddings)

        if labels is not None:
            loss_fcn = nn.CrossEntropyLoss()
            # print(logits.device, answers.device)
            loss = loss_fcn(logits, labels.view(-1))
            if torch.isnan(loss):
                print(logits, labels)
            return loss, logits               

        return logits

class ALBEFforTxtClassification(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))            
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        cls_config = config['cls_head']
        if cls_config['head_type'] == 'linear':
            self.cls_head = nn.Linear(text_width, cls_config['class_num'])
        elif cls_config['head_type'] == 'mlp':
            self.cls_head = nn.Sequential(
                    nn.Linear(text_width, text_width * cls_config['cls_hidden_scale']),
                    nn.ReLU(),
                    nn.Linear(text_width * cls_config['cls_hidden_scale'], cls_config['class_num'])
                )

    def model_part(self, part_name):
        tmp_config = self.text_encoder.config
        num_layers = tmp_config.num_hidden_layers
        fusion_layers = tmp_config.fusion_layer
        if part_name == 'txt_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(0, fusion_layers)] + [self.text_proj]
        elif part_name == 'vis_encoder':
            return [self.visual_encoder, self.vision_proj]
        elif part_name == 'embedding':
            return [self.text_encoder.embeddings]
        elif part_name == 'fusion_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(fusion_layers, num_layers)]
        elif part_name == 'task_head':
            return [self.cls_head]
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
    
    def forward(self, sent_ids, sent_attn_mask, cond_ids, cond_attn_mask, labels=None, alpha=0):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        # print('in model', sent_ids.shape)
        cond_output = self.text_encoder(cond_ids, attention_mask = cond_attn_mask,
                                        return_dict = True, mode = 'text')
        cond_embeds = cond_output.last_hidden_state
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
 

        text_output = self.text_encoder(sent_ids, attention_mask = sent_attn_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state

        # print(text_embeds.shape, image_embeds.shape, questions.attention_mask.shape, image_atts.shape)

        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = sent_attn_mask,
                                        encoder_hidden_states = cond_embeds,
                                        encoder_attention_mask = cond_attn_mask,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                                

        vl_embeddings = output_pos.last_hidden_state[:,0,:]
        logits = self.cls_head(vl_embeddings)

        if labels is not None:
            loss_fcn = nn.CrossEntropyLoss()
            # print(logits.device, answers.device)
            loss = loss_fcn(logits, labels.view(-1))
            if torch.isnan(loss):
                print(logits, labels)
            return loss, logits               

        return logits

class ALBEF_NLVR(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config.num_hidden_layers = 18
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )            

        self.share_cross_attention(self.text_encoder.encoder)
            
            
    def forward(self, image, text, targets):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))                   

        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]            
        prediction = self.cls_head(hidden_state)

        if targets is not None: 
            loss = F.cross_entropy(prediction, targets)     
            return loss, prediction  
        else:
            return prediction
                

    def share_cross_attention(self, model):
            
        for i in range(6):
            layer_num = 6+i*2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias 


class ALBEF_NLVR2(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)      
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size*2, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )            

        # self.share_cross_attention(self.text_encoder.encoder)
            
    def model_part(self, part_name):
        tmp_config = self.text_encoder.config
        num_layers = tmp_config.num_hidden_layers
        fusion_layers = tmp_config.fusion_layer
        if part_name == 'txt_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(0, fusion_layers)]
        elif part_name == 'vis_encoder':
            return [self.visual_encoder]
        elif part_name == 'embedding':
            return [self.text_encoder.embeddings]
        elif part_name == 'fusion_encoder':
            return [self.text_encoder.encoder.layer[i] for i in range(fusion_layers, num_layers)]
        elif part_name == 'task_head':
            return [self.cls_head]
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
    
    def forward(self, image1, image2, input_ids, attention_mask, targets):
        
        image = torch.cat([image1, image2], dim=0)
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))                   

        text_output = self.text_encoder(input_ids, attention_mask = attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        
        output0 = self.text_encoder(encoder_embeds = text_embeds, 
                                   attention_mask = attention_mask, 
                                   encoder_hidden_states = image0_embeds,
                                   encoder_attention_mask = image_atts[:image0_embeds.size(0)],        
                                   return_dict = True,
                                   mode = 'fusion'
                                  ) 
        output1 = self.text_encoder(encoder_embeds = text_embeds, 
                                   attention_mask = attention_mask, 
                                   encoder_hidden_states = image1_embeds,
                                   encoder_attention_mask = image_atts[image0_embeds.size(0):],        
                                   return_dict = True,
                                   mode = 'fusion'
                                  ) 
        hidden_state = torch.cat([output0.last_hidden_state[:,0,:], output1.last_hidden_state[:,0,:]], dim=1)            
        prediction = self.cls_head(hidden_state)

        if targets is not None: 
            loss = F.cross_entropy(prediction, targets)     
            return loss, prediction  
        else:
            return prediction