python ./finetune_table_retr.py \
    --do_train \
    --model_path ./pretrained_models/tqa_reader_base \
    --train_data ./data/auto_tune/sql_retr_data_train_small.jsonl \
    --eval_data ./data/auto_tune/sql_retr_data_dev_small.jsonl \
    --n_context 10 \
    --per_gpu_batch_size 2 \
    --cuda 0 \
    --name finetune_table_retr \
    --checkpoint_dir output \
    --checkpoint_steps 20

