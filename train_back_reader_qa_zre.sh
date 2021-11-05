python train_reader.py \
        --train_data ./open_domain_data/QA_ZRE/train.jsonl \
        --eval_data ./open_domain_data/QA_ZRE/dev.jsonl \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --name qa_zre_backward_reader \
        --checkpoint_dir output \
        --total_steps 30000 \
        --eval_freq 5000 \
        --save_freq 5000 \
        --backward 1 \

