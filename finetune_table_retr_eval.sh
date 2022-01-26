python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./output/finetune_table_retr_margin/epoc_0_step_24000_model.pt \
    --eval_data ./data/retrieved_test.json \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name retrieved_test_margin \
    --checkpoint_dir output \

