python ./test_table_retr.py \
        --model_path ./pretrained_models/tqa_reader_base \
        --eval_data /home/cc/code/open_table_discovery/dataset/nq_tables/bm25_template_graph/dev/fusion_in_decoder_data.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 200 \
        --name table_retr_tqa_dev_nq_tables_bm25 \
        --checkpoint_dir output \

