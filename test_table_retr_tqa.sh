dataset=nq_tables
expr=triple_template_graph
mode=dev
python ./test_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --eval_data /home/cc/code/open_table_discovery/dataset/${dataset}/${expr}/${mode}/fusion_in_decoder_data.jsonl \
    --per_gpu_batch_size 1 \
    --n_context 500 \
    --name ${dataset}_${expr}_${mode} \
    --checkpoint_dir output \

