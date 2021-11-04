python test_reader.py \
        --model_path ./pretrained_models/tqa_reader_large \
        --eval_data ./open_domain_data/TQA/test.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name tqa_test \
        --checkpoint_dir output \

