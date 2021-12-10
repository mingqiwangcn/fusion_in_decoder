python ./test_table_retr.py \
        --model_path ./pretrained_models/tqa_reader_base \
        --eval_data /home/cc/code/open_table_discovery/dataset/fetaqa/template_graph/dev/fusion_in_decoder_data.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 10000 \
        --name table_retr_tqa_test \
        --checkpoint_dir output \

