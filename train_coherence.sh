python train_coherence.py \
        --train_data ./open_domain_data/QA_ZRE/train_1_per.jsonl \
        --eval_data ./open_domain_data/QA_ZRE/dev_100.jsonl \
        --f_reader_model_path ./pretrained_models/qa_zre_forward_reader_data_1_percent/checkpoint/best_dev \
        --b_reader_model_path ./pretrained_models/qa_zre_backward_reader_data_1_percent/checkpoint/best_dev \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --name coherence_train_data_1_percent \
        --checkpoint_dir output \
        --total_steps 30000 \
        --eval_freq 1000 \
        --save_freq 1000 \

