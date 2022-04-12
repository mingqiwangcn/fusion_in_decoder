if [ "$#" -ne 3 ]; then
    echo "Usage: ./finetune_syt_retr_.sh <dataset> <experiment> <sql_expr>"
    exit
fi
dataset=$1
exprt=$2
sql_expr=$3
exprt_dir=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}/${exprt}
chk_name=train_syt_${dataset}_${exprt}_${sql_expr}_tagged
python ./finetune_table_retr.py \
    --do_train \
    --model_path ./pretrained_models/tqa_reader_base \
    --train_data ${exprt_dir}/fusion_retrieved_train.jsonl \
    --eval_data ${exprt_dir}/fusion_retrieved_dev.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name ${chk_name} \
    --checkpoint_dir output \
    --checkpoint_steps 1000 \
    --retr_model_type general
