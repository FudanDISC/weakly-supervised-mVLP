from transformers import XLMRobertaForMaskedLM
import torch
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--albef_ckpt', type=str, default=None, help='the albef checkpoint')
    parser.add_argument('--xlm_ckpt', type=str, default='xlm-roberta-base', help='the xlmr checkpoint, local or from huggingface')
    parser.add_argument('--output', type=str, default=None, help='the target output checkpoint path')
    args = parser.parse_args()

    print('loading xlm-r checkpoint from {}'.format(args.xlm_ckpt))
    xlm_model = XLMRobertaForMaskedLM.from_pretrained(args.xlm_ckpt)
    xlm_ckpt = xlm_model.state_dict()

    print('loading albef checkpoint from {}'.format(args.albef_ckpt))
    albef_ckpt = torch.load(args.albef_ckpt, map_location='cpu')
    if 'model' in albef_ckpt:
        albef_ckpt = albef_ckpt['model']
    keys_to_del = [k for k in albef_ckpt if k.split('.')[0].endswith('_m') or 'queue' in k]
    for k in keys_to_del:
        del(albef_ckpt[k])

    reframed_keys = []
    for k,v in xlm_ckpt.items():
        if k.startswith('roberta.'):
            new_k = 'text_encoder.bert.{}'.format(k[8:])
            try:
                layer_num = int(k.split('.')[3])
            except:
                assert new_k in albef_ckpt, '{} key error!'.format(new_k)
                reframed_keys.append((k, new_k))
                continue
            if 'attention' in k and layer_num > 5:
                tmp_k = re.sub('attention', 'crossattention', new_k)
                # assert tmp_k in albef_ckpt, '{} key error!'.format(tmp_k)
                reframed_keys.append((k, tmp_k))
        elif k.startswith('lm_head'):
            lm_head_map = {'lm_head.bias': 'text_encoder.cls.predictions.bias',
                            'lm_head.dense.weight': 'text_encoder.cls.predictions.transform.dense.weight',
                            'lm_head.dense.bias': 'text_encoder.cls.predictions.transform.dense.bias',
                            'lm_head.layer_norm.weight': 'text_encoder.cls.predictions.transform.LayerNorm.weight', 
                            'lm_head.layer_norm.bias': 'text_encoder.cls.predictions.transform.LayerNorm.bias', 
                            'lm_head.decoder.weight': 'text_encoder.cls.predictions.decoder.weight', 
                            'lm_head.decoder.bias': 'text_encoder.cls.predictions.decoder.bias'}
            new_k = lm_head_map[k]
        else:
            raise ValueError
        
        assert new_k in albef_ckpt, '{} key error!'.format(new_k)
        reframed_keys.append((k, new_k))

    for k_pair in reframed_keys:
        old_k, new_k = k_pair
        albef_ckpt[new_k] = xlm_ckpt[old_k]
    
    print('saving the result checkpoint to {}'.format(args.output))
    torch.save(albef_ckpt, args.output)

if __name__=='__main__':
    main()

    


