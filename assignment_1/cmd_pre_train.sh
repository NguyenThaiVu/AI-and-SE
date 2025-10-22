nohup python pretrain_t5_span_corruption.py \
  --tokenizer_dir tokenizer_t5_code \
  --output_dir t5_code_pretrained \
  --base_model t5-small \
  > pretrain_t5.log 2>&1 &