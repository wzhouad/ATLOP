python train.py --data_dir ./dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--log_name label-semantic-htr-interact \
--save_path checkpoints/6-roberta-model-htr-interact.pt \
--save_last checkpoints/6-roberta-model-last-htr-interact.pt \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--relinfo_file rel_info.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 66 \
--num_class 97
