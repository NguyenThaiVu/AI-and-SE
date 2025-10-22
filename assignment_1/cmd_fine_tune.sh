nohup python fine_tune_t5_if_condition.py \
  --tokenizer_dir tokenizer_t5_code \
  --model_dir t5_code_pretrained \
  > finetune_t5.log 2>&1 &