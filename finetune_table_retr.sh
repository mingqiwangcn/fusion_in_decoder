python -m pdb ./finetune_table_retr.py \
    --do_train \
    --model_path ./pretrained_models/tqa_reader_base \
    --train_data ./data/auto_tune/sql_retr_data_train.jsonl \
    --eval_data ./data/retrieved/retrieved_dev.json \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name finetune_table_retr_adapt \
    --checkpoint_dir output \
    --checkpoint_steps 3000

