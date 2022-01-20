dataset=nq_tables
expr=triple_template_graph
mode=test_nq
python ./test_table_retr.py \
    --model_path ./pretrained_models/nq_reader_base \
    --eval_data ./data/retrieved_data_test.json \
    --per_gpu_batch_size 1 \
    --n_context 500 \
    --name ${dataset}_${expr}_${mode} \
    --checkpoint_dir output \

