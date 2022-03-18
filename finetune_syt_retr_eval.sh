if [ "$#" -ne 5 ]; then
    echo "Usage: ./finetune_syt_retr_eval.sh <dataset> <mode> <expr> <epoch> <step>"
    exit
fi
dataset=$1
mode=$2
expr=$3
epoch=$4
step=$5
data_dir=~/code/open_table_discovery/table2txt/dataset/${dataset}/${expr}
python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./pretrained_models/train_syt_${dataset}_${expr}/epoc_${epoch}_step_${step}_model.pt \
    --eval_data ${data_dir}/fusion_retrieved_${mode}.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name syt_${mode}_${dataset}_${expr}_step_${step} \
    --checkpoint_dir output \
