# Generic Dependency Modeling for Multi-party Conversations


## Quick Links

- [Generic Dependency Modeling for Multi-party Conversations](#generic-dependency-modeling-for-multi-party-conversations)
  - [Quick Links](#quick-links)
  - [Requirements](#requirements)
  - [Description of Codes](#description-of-codes)
  - [Preprocessing](#preprocessing)
  - [How to run](#how-to-run)
    - [Pretrain](#pretrain)
    - [ERC](#erc)
    - [Dialog-MRC](#dialog-mrc)
    - [Dialog-RE](#dialog-re)
    - [Dialog-Summarization](#dialog-summarization)

## Requirements

- torch>=1.7
- transformers==4.8.1
- datasets==1.12.1
- nltk==3.6.3
- rouge-score

## Description of Codes
- `./pretrain` -> directories for the pretraining codes.
- `./ERC` -> codes for applying ReDE in ERC tasks like MELD.
- `./MRC` -> codes for applying ReDE in Dialogue machine reading comprehension tasks like Molweni-MRC.
- `./RE` -> codes for applying ReDE in Dialogue relation extraction tasks like DialogRE.
- `./SUM` -> codes for applying ReDE in Dialogue summarization tasks like SAMSum.

## Preprocessing

We are preparing the data collecting and preprocessing codes for the MELD, Molweni-MRC, DialogRE and SAMSum. So far you can access the full preprocessed data by contacting [shenwzh3@mail2.sysu.edu.cn](shenwzh3@mail2.sysu.edu.cn). We also provide the subset of data in this repo.

## How to run

### Pretrain

To pretrain RoBERTa, first run the pre-tokenize scripts:
```
python pretrain/roberta/prepro.py
```
Then run the training scripts:
```
EXPORT output_path=saves/pretrain 
EXPORT model_path=roberta-large

python -m torch.distributed.launch --nproc_per_node 8 pretrain/roberta/run_mlm.py \
--overwrite_output_dir \
--do_train True --do_eval True --do_predict True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-5 \
--num_train_epochs 5 \
--evaluation_strategy steps\
--eval_steps 1000 \
--logging_strategy steps \
--logging_steps 1000 \
--save_strategy steps \
--save_steps 1000 \
--load_best_model_at_end True \
--save_total_limit 20 \
--max_seq_length 512 \
--max_hop 100 \
--relative_mode bi \
--output_dir $output_path \
--model_name_or_path $model_path \
--config_name $model_path \
--tokenizer_name  $model_path \
--dataset_name all \
--update_partial

```


To pretrain BART, first run the pre-tokenize scripts:
```
python pretrain/bart/prepro.py
```
Then run the training scripts:
```
EXPORT output_path=saves/pretrain 
EXPORT model_path=bart-large
python -m torch.distributed.launch --nproc_per_node 8 pretrain/bart/run_lm.py \
    --overwrite_output_dir \
    --do_train --do_eval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --save_total_limit 20 \
    --evaluation_strategy steps --eval_steps 1000 \
    --logging_strategy steps --logging_steps 1000 \
    --save_strategy steps --save_steps 1000 \
    --load_best_model_at_end True \
    --generation_num_beams 4 \
    --model_name_or_path $model_path \
    --config_name $model_path \
    --tokenizer_name $model_path \
    --max_source_length 1024 --max_target_length 1024 --generation_max_length 1024\
    --max_hop 100 --relative_mode bi\
    --output_dir $output_path \
    --dataset_name all \
    --prediction_loss_only \
    --update_mode partial
```


### ERC
Run:
```
EXPORT model_path=roberta-large

python -m torch.distributed.launch --nproc_per_node 4 REC/run.py \
--lr 1e-5 --grad_accumulate_step 2 --max_sent_len 200 \
--batch_size 4 --max_hop 100 --dropout 0.2\
--bert_model_dir $model_path \
--bert_tokenizer_dir $model_path \
--data_dir ./data/
```

### Dialog-MRC

Run:

```
EXPORT model_path=roberta-large
EXPORT output_path=saves/mrc

python -m torch.distributed.launch --nproc_per_node 8 MRC/run_qa.py \
    --overwrite_output_dir \
    --do_train True --do_eval True --do_predict True \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 --num_train_epochs 20 \
    --evaluation_strategy epoch --logging_strategy epoch --save_strategy epoch \
    --load_best_model_at_end True --dataloader_pin_memory True \
    --model_name_or_path $model_path \
    --config_name $model_path \
    --tokenizer_name $model_path \
    --max_hop 100 --relative_mode bi \
    --output_dir $output_path \
    --data_path data/

```

### Dialog-RE

Run:

```
EXPORT model_path=roberta-large
python -m torch.distributed.launch --nproc_per_node 4 RE/run.py \
      --lr 5e-6 --grad_accumulate_step 2 --max_sent_len 400 \
      --batch_size 4 --max_hop 100 --dropout 0.2\
      --bert_model_dir $model_path \
      --bert_tokenizer_dir $model_path \
      --data_dir data/

```


### Dialog-Summarization

Run:

```

EXPORT model_path=bart-large
EXPORT output_path=saves/sum
python -m torch.distributed.launch --nproc_per_node 4 SUM/run_summarization.py \
      --overwrite_output_dir \
      --do_train True --do_eval True --do_predict True  \
      --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 2 \
      --learning_rate 1e-5 --num_train_epochs 20 \
      --evaluation_strategy epoch --logging_strategy epoch --save_strategy epoch \
      --load_best_model_at_end True --metric_for_best_model rouge2 \
      --dataloader_pin_memory True \
      --predict_with_generate True --generation_max_length 100 \
      --generation_num_beams 4 \
      --model_name_or_path $model_path \
      --config_name $model_path \
      --tokenizer_name $model_path \
      --max_source_length 600 --max_target_length 100 \
      --max_hop 100 --relative_mode bi --disable_tqdm False\
      --output_dir $output_path \
      --data_path data/ 
```

