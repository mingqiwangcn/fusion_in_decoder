if [ "$#" -ne 5 ]; then
    echo "Usage: ./finetune_syt_retr_.sh <dataset> <experiment> <sql_expr> <train itr> <part_no>"
    exit
fi
dataset=$1
exprt=$2
sql_expr=$3
train_itr=$4
part_no=$5
exprt_dir=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}
chk_name=${dataset}_${exprt}_${sql_expr}_${train_itr}_part_${part_no}
python ./finetune_table_retr.py \
    --do_train \
    --model_path ~/code/models/tqa_reader_base \
    --train_data ${exprt_dir}/${train_itr}/${exprt}/data_parts/part_${part_no}.jsonl \
    --eval_data ${exprt_dir}/dev/${exprt}/fusion_retrieved_tagged.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 4 \
    --cuda 0 \
    --name ${chk_name} \
    --checkpoint_dir output \
    --max_epoch 10 \
    --ckp_num 2 \
    --patience_steps 6 \
    --question_maxlength 50 \
    --text_maxlength 300 \
