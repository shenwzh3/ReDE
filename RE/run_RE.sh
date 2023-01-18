#!/bin/bash

lr_candidate=(1e-5 5e-5 1e-4)
gradient_accumulation_steps_candidate=(1 2)
dropout_candidate=(0.0 0.1)

for dropout in ${dropout_candidate[*]}
do
  for lr in ${lr_candidate[*]}
  do
    for grad_accumulate_step in ${gradient_accumulation_steps_candidate[*]}
    do
      CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 run.py \
      --lr "${lr}" --grad_accumulate_step "${grad_accumulate_step}" --max_sent_len 400 \
      --batch_size 4 --max_hop 100 --dropout "${dropout}"\
      --bert_model_dir xxx \
      --bert_tokenizer_dir xxx \
      --data_dir ./data/ 

    done
  done
done