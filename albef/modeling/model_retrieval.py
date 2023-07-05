'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertForMaskedLM, BertModel

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
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

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])    
        self.itm_head = nn.Linear(text_width, 2)
        self.forward_mod = 'train'

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
            return [self.itm_head]
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

    def forward(self, image, text_input_ids, text_attention_mask, target_input_ids=None, target_attn_mask=None, forward_mod='multimodal'):
        if forward_mod == 'txt':
            return self.forward_txt_trans(text_input_ids, text_attention_mask, target_input_ids, target_attn_mask)
        if self.forward_mod == 'train':
            return self.forward_train(image, text_input_ids, text_attention_mask)
        elif self.forward_mod == 'coarse':
            return self.forward_coarse(image, text_input_ids, text_attention_mask)
        elif self.forward_mod == 'fine':
            return self.forward_fine(image, text_input_ids, text_attention_mask)
        else:
            raise NotImplementedError     

    def forward_txt_trans(self, text_input_ids, text_attention_mask, text_trans_input_ids, text_trans_attention_mask):
        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        trans_text_output = self.text_encoder(text_trans_input_ids, attention_mask = text_trans_attention_mask,                      
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
    
    def forward_train(self, image, text_input_ids, text_attention_mask, alpha=0):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        # print('in model', text_input_ids.shape)

        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                       

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = sim_i2t.t()

        # sim_targets = torch.zeros(sim_i2t.size()).to(sim_i2t.device).to(sim_i2t.dtype)
        sim_targets = torch.zeros_like(sim_i2t)
        sim_targets.fill_diagonal_(1)
        # print(sim_i2t.shape)
        # print(sim_targets.shape)
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text_attention_mask,
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
        try:
            hard_img_index = torch.multinomial(weights_t2i, 1).squeeze()
        except:
            print(sim_t2i)
            torch.save(sim_t2i, '/remote-home/zjli/tmp_debug/sim_mat.pth')
            raise ValueError
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
            print(sim_i2t)
            torch.save(sim_i2t, '/remote-home/zjli/tmp_debug/sim_mat.pth')
            raise ValueError
        text_embeds_neg = torch.index_select(text_embeds, dim=0, index=hard_text_index)
        text_atts_neg = torch.index_select(text_attention_mask, dim=0, index=hard_text_index)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(encoder_embeds = text_embeds_all, 
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
        loss = loss_ita + loss_itm         

        return loss, vl_output, loss_ita, loss_itm, itm_labels

    def forward_fine(self, image, text_input_ids, text_attention_mask, alpha=0):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
 

        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state

        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text_attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                                

        vl_embeddings = output_pos.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings)               

        return vl_output

    def forward_coarse(self, image=None, text_input_ids=None, text_attention_mask=None):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        if image is not None:
            image_embeds = self.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        else:
            image_feat = None  

        if text_input_ids is not None:
            text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)             
        else:
            text_feat = None          
        return image_feat, text_feat


        

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
