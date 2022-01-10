python -m pdb ./train_passage_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --eval_data /home/cc/data/qa_zre_data/data/open_qa/dev/qa_zre_fushion_input.jsonl \
    --per_gpu_batch_size 1 \
    --n_context 30 \
    --name train_passage_retr \
    --checkpoint_dir output \

