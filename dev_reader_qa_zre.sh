python -m pdb test_reader.py \
        --model_path ./pretrained_models/qa_zre_forward_reader/checkpoint/best_dev \
        --eval_data ./open_domain_data/QA_ZRE/dev.jsonl \
        --per_gpu_batch_size 3 \
        --n_context 10 \
        --name qa_zre_forward_dev \
        --checkpoint_dir output \

