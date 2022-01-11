python ./train_passage_retr.py \
    --do_train \
    --model_path ./pretrained_models/tqa_reader_base \
    --train_data /home/cc/data/qa_zre_data/data/open_qa/train/data_5_percent/qa_zre_fusion_train.jsonl \
    --train_qas /home/cc/data/qa_zre_data/data/open_qa/train/data_5_percent/qa_zre_train_5_percent_qas.json \
    --eval_data /home/cc/data/qa_zre_data/data/open_qa/dev/qa_zre_fusion_dev.jsonl \
    --eval_qas /home/cc/data/qa_zre_data/data/open_qa/dev/qa_zre_dev_100_qas.json \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name train_passage_retr \
    --checkpoint_dir output \

