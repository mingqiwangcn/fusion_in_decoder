step=$1
python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./output/finetune_table_retr_margin/epoc_0_step_${step}_model.pt \
    --eval_data ./data/auto_tune/sql_retr_data_dev.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name auto_tune_dev_${step} \
    --checkpoint_dir output \

