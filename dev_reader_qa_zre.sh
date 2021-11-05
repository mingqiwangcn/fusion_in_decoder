python test_reader.py \
        --model_path ./output/qa_zre_foward_reader/checkpoint/best_dev \
        --eval_data ./open_domain_data/QA_ZRE/dev.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name qa_zre_dev \
        --checkpoint_dir output \

