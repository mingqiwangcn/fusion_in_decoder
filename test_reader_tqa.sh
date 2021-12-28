python test_reader.py \
        --model_path ./pretrained_models/tqa_reader_base \
        --eval_data ~/data/TQA/dev.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name tqa_dev \
        --checkpoint_dir output \

