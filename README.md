# MVPTR: Multi-Stage Vision-Language Pre-Training via Multi-Level Semantic Alignment

## Introduction

This repository is the implementation of our project [MVPTR: Multi-Stage Vision-Language Pre-Training via Multi-Level Semantic Alignment](https://arxiv.org/abs/2201.12596). In this paper, we propose to explicitly align vision and language at multiple levels. In MVP, we firstly introduce concepts in both modalities to construct two-level semantic representations for language and vision, then we design a 2-stage pre-training framework to learn intra-modal and cross-modal interaction respectively. The procedure is illustrated in the figure below:

![MVPTR](./figs/MVP.png)

Part of the implementation is based on the project [Oscar&VinVL](https://github.com/microsoft/Oscar) and [ALBEF](https://github.com/salesforce/ALBEF), many thanks to Microsoft and Salesforce for the open-source resource.

## Installation

```bash
# create environment
conda create --name weak_mvlp python=3.9
conda activate weak_mvlp

# install pytorch (you can install the version that fit your GPU! >=1.8.0 is recommended!)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# run setup
python setup.py build develop

# install requirements
pip install -r requirements.txt

mkdir pretrained_models
mkdir pretrain_datasets
```

### Data Preprocess:

To transform image-text pairs to fit the MVP input, there are several steps: 

1. We utilize the off-the-shelf scene graph parser provided in [SPICE](https://github.com/peteanderson80/SPICE) to extract tuples from text, which are considered as phrase concepts.
2. Only general phrases which appear in at least 50 sentences in the pre-training corpus are considered, the id2phrase mapping is stored in [id2phrase](). Extracted phrases for coco, Flickr, vqa, visual entailment, referring expression can be downloaded from [here](https://drive.google.com/file/d/1fl8vXvxw9ZXmLaQ6a_4KrM-Zi1w_PgwN/view?usp=sharing).
3. Object features and tags are extracted from images with [the object detector used in VinVL](https://github.com/microsoft/scene_graph_benchmark).

To download extracted features used in VinVL and MVP, it is recommended to use [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).

## Usage

For the convenience of usage, we considered three different scenarios:
- For a quick start to use MVPTR, we provide a complete pipeline which takes raw image-text pairs as inputs, please check the Quick Start section.
- To utilize MVPTR on the already-implemented tasks (image text retrieval, vqa, SNLI-VE, Referring Expression), please check the  section that corresponds to the task and dataset.
- To fine-tune MVPTR on datasets that are not considered in our experiments, please check the Fine-Tuning on Custom Datasets section.

### Prepare the Tools

If you are willing to use the pipeline or fine-tune on your custom data, several tools need to be prepared before you start:

1. The object detector used to extract region features, we provide a torchscripted version of VinVL, which is a simple .pt file and can be directly used by:

   ```python
   import torchvision, torch
   od_model = torch.jit.load('OD_MODAL_PATH')
   ```

   You can download it from (to be uploaded, coming soon)

2. The SPICE parser to extract phrases from texts, we provide a script to help you:

   ```bash
   bash tools/prepare_spice.sh
   ```

   
### Quick Start

We provide a complete pipeline of MVPTR which performs masked language model or generates contextual representations directly from the image-text input, it can be used only for inference (the encoded representations can be freezed and used for your own tasks).

You can quickly test MVPTR, first download [pretrained_base](https://drive.google.com/file/d/1t9S9ejwpit7UO_Y5f9pYnn7p9g-AtWtC/view?usp=sharing) to pretrained_mdoels/base and use the mlm pipeline:

```python
from oscar.modeling.modeling_pipeline import InferencePipeline

pipeline = InferencePipeline(model_name='mlm', model_path='pretrained_models/base/', 'OD_MODEL_PATH', od_config_dir='tools/configs/', parser_path='tools/spice', id2phrase='datasets/mvp/id2phrase_new.json')
print(inference.inference('figs/coco_test.jpg', 'two [MASK] are playing on a ground'))
```

Based on the image, our model generates the "dogs" output:

![coco_test](./figs/coco_test.jpg)

If you change the model_name from "mlm" to "embedding", the pipeline generates the contextual representations of all tokens, phrase concepts and regions, which can be used for downstream task (both cross-modal output and uni-modal output).

Notice that it is not recommended to use the pipeline for fine-tuning, but you can refer to the implementation of the pipeline to better understand how to use MVPTR. If you need to fine-tune the MVPTR, you need to first extract and regions features and phrases, please refer to the sub-sections below.

### Image-Text Retrieval

#### Flickr30k

As the original pre-trained VinVL uses the test split of Flickr30k during pre-training, we exclude those data and pre-train our MVP to avoid the information leaking:

1. Download the pre-trained checkpoint for flickr30k retrieval: [mvp_base_for_fk](https://drive.google.com/file/d/19L2ej4s-fhm5qCQzI2g3pxgVIrdd9FeV/view?usp=sharing)

2. Download the extracted features and captions from: [fk_ir](https://drive.google.com/file/d/1L4xXrk3q0e6DsBiucJkUFWbxqPLtzRjY/view?usp=sharing)

3. Training with evaluation on the validation split:

```bash
python3 oscar/run_retrieval.py \
    --model_name_or_path pretrained_models/base_for_fk/ \
    --data_dir datasets/fk_ir --do_train --do_lower_case \
    --per_gpu_train_batch_size 64 --learning_rate 0.00004 \
    --num_train_epochs 10 --weight_decay 0.05 \
    --save_steps 400 --add_od_labels --od_label_type vg \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
    --max_img_seq_length 50  --dataset_name flickr \
    --max_phrases 5  --sent_sg datasets/fk_ir/fk_sg.pt \
    --id2node datasets/mvp/id2phrase_new.json \
    --output_dir output_fk/mvp_std/
```

4. Evaluation on the test split:

```bash
python3 oscar/run_retrieval.py \
    --eval_model_dir output_fk/mvp_std/checkpoint-x-x \
    --data_dir datasets/fk_ir --do_test --do_eval --do_lower_case \
    --per_gpu_eval_batch_size 128 --add_od_labels --od_label_type vg \
    --max_seq_length 50  --test_split test \
    --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
    --max_img_seq_length 50  --dataset_name flickr \
    --max_phrases 5  --sent_sg datasets/fk_ir/fk_sg.pt \
    --id2node datasets/mvp/id2phrase_new.json \
    --output_dir output_fk/mvp_std/
```

We found that MVP quickly converges to the best performance within 2 epochs.

#### MSCOCO

1. Download the tsv-format features: 

   ```bash
   path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/' pretrain_datasets/coco --recursive
   ```

2. Download the captions:

   ```bash
   path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_ir' datasets/ --recursive
   ```

3. Download the pre-trained checkpoint: [mvp_base_for_coco](https://drive.google.com/file/d/1VUl9TbtkKYB8BDc9Xk-a9G0XYTnyUwpz/view?usp=sharing)

4. Training with evaluation on the 1k minival split:

   ```bash
   python3 oscar/run_retrieval.py \
       --model_name_or_path pretrained_models/base_for_coco/ \
       --per_gpu_train_batch_size 96 --learning_rate 0.00004 \
       --num_train_epochs 15 --weight_decay 0.05 \
       --save_steps 400 --add_od_labels --od_label_type vg \
       --max_seq_length 50  --evaluate_during_training \
       --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
       --max_img_seq_length 50 \
       --id2node datasets/mvp/id2phrase_new.json \
       --do_train  --img_feat_file pretrain_datasets/coco/model_0060000/features.tsv \
       --do_lower_case  --data_dir datasets/coco_ir/ \
       --evaluate_during_training  --max_tag_length 30 \
       --output_dir output_coco/mvp --dataset_name coco \
       --max_phrases 5  --sent_sg datasets/coco_ir/coco_full_sg.pt
   ```

4. Evaluationation on the 5k test split:

   ```bash
   python3 oscar/run_retrieval.py \
       --eval_model_dir output_coco/mvp/checkpoint-x-x \
       --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
       --test_split test  --eval_img_keys_file test_img_keys.tsv \
       --id2node datasets/mvp/id2phrase_new.json --do_test --do_eval \
       --img_feat_file pretrain_datasets/coco/model_0060000/features.tsv \
       --do_lower_case  --data_dir datasets/coco_ir/ \
       --output_dir evaluate_coco/mvp --dataset_name coco \
       --max_phrases 5  --sent_sg datasets/coco_ir/coco_full_sg.pt
   ```

### VQA

1. In VQA, we found it useful to initialize the classifier head with the weights used in MLM task, the initialized checkpoint can be downloaded from: [mvp_base_for_vqa](https://drive.google.com/file/d/1QsUHakD1tEip7txe7BUiMgDcnI1AtOva/view?usp=sharing)

2. Download vqa dataset:

   ```bash
   path/to/azcopy copy "https://biglmdiag.blob.core.windows.net/vinvl/datasets/vqa" datasets/ --recursive
   ```

   and we re-use the tsv-format coco image features mentioned in MSCOCO retrieval.

3. To achieve the best performance, the model is trained on train and val splits:

   ```bash
   python3 oscar/run_vqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length 50 \
       --data_label_type mask --img_feature_type faster_r-cnn --data_dir datasets/vqa/ \
       --model_type bert --model_name_or_path pretrained_models/base_for_vqa/ \
       --task_name vqa_text --do_train_val --do_lower_case \
       --max_seq_length 128 --per_gpu_eval_batch_size 256 \
       --per_gpu_train_batch_size 32 --learning_rate 5e-05 \
       --num_train_epochs 25 --add_od_labels \
       --output_dir output_vqa/mvp/ --label_file datasets/vqa/trainval_ans2label.pkl \
       --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 20 --eval_step 4000 --drop_out 0.3 \
       --weight_decay 0.05 --warmup_steps 0 --loss_type bce \
       --img_feat_format tsv --img_feat_dir pretrain_datasets/coco/model_0060000/ \
       --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/vqa \
       --max_tag_length 30 --use_pretrain --use_b_text --b_as_list
   ```

4. Predict the answers:

   ```bash
   python3 oscar/run_vqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length 50 \
       --data_label_type mask --img_feature_type faster_r-cnn --data_dir datasets/vqa/ \
       --model_type bert --model_name_or_path pretrained_models/base_for_vqa/ \
       --task_name vqa_text --do_test --do_lower_case \
       --max_seq_length 128 --per_gpu_eval_batch_size 256 \
       --label2ans_file datasets/vqa/trainval_label2ans.pkl  --add_od_labels \
       --output_dir output_vqa/mvp/checkpoint-5-15120/ \
       --label_file datasets/vqa/trainval_ans2label.pkl --img_feat_format tsv \
       --img_feat_dir pretrain_datasets/coco/model_0060000/coco2015test/ \
       --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/vqa \
       --max_tag_length 30  --use_pretrain  --use_b_text --b_as_list
   ```

Here is the detailed evaluation results from the VQA challange on EvalAI:

```json
[{"test-dev": {"yes/no": 91.55, "number": 58.47, "other": 67.04, "overall": 76.16}}, {"test-standard": {"yes/no": 91.65, "number": 58.45, "other": 67.16, "overall": 76.36}}]
```

### SNLI-VE

1. Download the dataset: coming soon.

2. Download the pre-trained checkpoint: [mvp_base](https://drive.google.com/file/d/1t9S9ejwpit7UO_Y5f9pYnn7p9g-AtWtC/view?usp=sharing)

3. Training:

   ```bash
   python3 oscar/run_ve.py -j 4 --img_feature_dim 2054 --max_img_seq_length 70 \
       --img_feature_type faster_r-cnn --data_dir datasets/ve/ --output_dir output_ve/mvp/ \
       --model_type bert --model_name_or_path pretrained_models/base/ \
       --task_name ve --do_train --do_lower_case --max_seq_length 70 \
       --per_gpu_eval_batch_size 128 --per_gpu_train_batch_size 64 \
       --learning_rate 4e-05 --num_train_epochs 25 --add_od_labels \
       --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 20 --eval_step 400 --drop_out 0.1 \
       --weight_decay 0.05 --warmup_steps 0 --loss_type ce \
       --img_feat_format pt --img_feat_dir datasets/ve/ \
       --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/ve \
       --max_tag_length 20 --bivinvl_id2phrase datasets/mvp/id2phrase_new.json
   ```
   
4. Evaluation:

   ```bash
   python3 oscar/run_ve.py -j 4 --img_feature_dim 2054 --max_img_seq_length 70 \
       --img_feature_type faster_r-cnn --data_dir datasets/ve/ \
       --output_dir output_ve/mvp/checkpoint-0-1035 \
       --model_type bert --model_name_or_path pretrained_models/base/ \
       --task_name ve --do_test --do_lower_case --max_seq_length 70 \
       --per_gpu_eval_batch_size 128 --add_od_labels \
       --img_feat_format pt --img_feat_dir datasets/ve/ \
       --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/ve \
       --max_tag_length 20 --bivinvl_id2phrase datasets/mvp/id2phrase_new.json
   ```

### Referring Expression

1. Download the dataset: [re](https://drive.google.com/file/d/1DmDM351JQdaTbS-WwU7VtxwJX5QitKUY/view?usp=sharing), we also re-use the tsv-format coco data.

2. Download the pre-trained checkpoint: [mvp_base](https://drive.google.com/file/d/1t9S9ejwpit7UO_Y5f9pYnn7p9g-AtWtC/view?usp=sharing)

3. Training:

   ```bash
   python3 oscar/run_re.py -j 4 --img_feature_dim 2054 --max_img_seq_length 50 \
       --img_feature_type faster_r-cnn --data_dir datasets/re/ --output_dir output_re/mvp/ \
       --model_type bert --model_name_or_path pretrained_models/base/ \
       --task_name re --do_train --do_lower_case --max_seq_length 30 \
       --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 64 \
       --learning_rate 4e-05 --num_train_epochs 25 --add_od_labels \
       --save_epoch 1 --seed 88 --evaluate_during_training \
       --logging_steps 20 --eval_step 1000 --drop_out 0.1 \
       --weight_decay 0.05 --warmup_steps 0 --img_feat_format tsv \
       --img_feat_dir pretrain_datasets/coco/model_0060000/ \
       --classifier linear --cls_hidden_scale 3 --txt_data_dir datasets/re \
       --max_tag_length 20  --loss_mod 1 \
       --bivinvl_id2phrase datasets/mvp/id2phrase_new.json \
       --data_file datasets/re/ve_splits.json --phrase_layer 2
   ```

### Pre-Training

1. Prepare the datasets:
   - Prepare the English image-text pairs: follow the configure in [pretrain_datasets/cc_coco_vg_img.yaml](https://github.com/FudanDISC/weakly-supervised-mVLP/blob/master/pretrain_datasets/cc_coco_vg_img.yaml), the corpus json file can be adopted from [here](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip), then you should donwload the corresponding images and set the path in this configure file.
   - Prepare the parallel sentence pairs: follow the configure in pretrain_datasets/wikimatrix_simplified.yaml, wikimatrix can be downloaded from [here](https://opus.nlpl.eu/WikiMatrix.php), we use all parallel corpus between English and target languages (make sure your dataset path can be loaded by the "load_from_disk" method provided by Huggingface.)
   - Prepare the unpaired multilingual corpus: follow the configure in, we use a sub-sampled [cc100](https://data.statmt.org/cc-100/) corpus, the dataset can also be downloaded from huggingface, the preprocess details are in our paper (make sure your dataset path can be loaded by the "load_from_disk" method provided by Huggingface.)

2. Prepare the initial checkpoint: download the ALBEF checkpoint from https://github.com/salesforce/ALBEF, then runs the following comman to get the initialized checkpoint INIT_CKPT:
   ```bash
   python utils/initialize_ckpt.py \
   --albef_ckpt ALBEF.pth \
   --xlm_ckpt xlm-roberta-base \
   --output INIT_CKPT
   ```

3. Training: run the following command to perform the unified pre-training from the INIT_CKPT!

   ```bash
   deepspeed --include localhost:0,1,2,3,4,5,6,7 cvlm/run_uni_stage2_albef.py \
    --deepspeed_config oscar/tmp_config.json --albef_config albef/configs/pretrain_base_xlm-r_freeze_vis.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/unified_mvlp/  \
    --tokenizer_name xlm-roberta-base --model_name_or_path INIT_CKPT \
    --do_lower_case --learning_rate 1e-04  --do_train  --mask_prob 0.15 --deepspeed  --avoid_mlm_head_tie \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 512 \
    --txt_dataset_file wikimatrix_simplified.yaml --train_batch_size_txt 2048  --img_txt_mod img-txt-full \
    --ckpt_period 8000 --max_iters 240000 --warmup_steps 24000   --log_period 10  --txt_txt_mod para_txt_full \
    --data_dir ./pretrain_datasets/ --dataset_file cc_coco_vg_img.yaml  --txt_dataformat transformers \
    --mono_txt_dataset_file text_mono/root_cc100_sub800M.yaml  --train_batch_size_mono_txt 2048  --mono_txt_max_length 64
    ```
The command above fits a server with 8 3090 GPUs, you can modify it at your wish.
```
