if [ "$#" -ne 5 ]; then
    echo "Usage: ./finetune_syt_retr_eval_bm25.sh <dataset> <mode> <expr> <epoch> <step>"
    exit
fi
dataset=$1
mode=$2
expr=$3
epoch=$4
step=$5
data_dir=~/code/open_table_discovery/table2txt/dataset/${dataset}/bm25_${expr}/${mode}
python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./output/bm25_train_syt_${dataset}_${expr}/epoc_${epoch}_step_${step}_model.pt \
    --eval_data ${data_dir}/bm25_fusion_retrieve_${mode}.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name bm25_syt_${mode}_${dataset}_${expr}_step_${step} \
    --checkpoint_dir output \
    --retr_model_type general \
