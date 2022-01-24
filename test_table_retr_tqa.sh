dataset=nq_tables
expr=auto_tune
mode=dev
python ./test_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --eval_data ./data/sql_retr_data_dev.jsonl \
    --per_gpu_batch_size 1 \
    --n_context 100 \
    --name ${dataset}_${expr}_${mode} \
    --checkpoint_dir output \

