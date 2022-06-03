if [ "$#" -ne 5 ]; then
    echo "Usage: ./retr_passages.sh <mode> <dataset> <experiment> <passage_file> <part>"
    exit
fi
mode=$1
dataset=$2
exptr=$3
passage_file_name=$4
part=$5
index_file=./data/on_disk_index_${dataset}_${exptr}/populated.index
exptr_dir=/home/cc/code/open_table_discovery/table2txt/dataset/${dataset}/${exptr}
passage_file=./data/on_disk_index_${dataset}_${exptr}/${passage_file_name}
query_file=/home/cc/data/${dataset}/fusion_query/fusion_query_${mode}_${part}.jsonl
out_retr_file=${exptr_dir}/fusion_retrieved_${mode}_${part}.jsonl

python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ${index_file} \
    --passage_file ${passage_file} \
    --data ${query_file} \
    --output_path ${out_retr_file} \
    --n-docs 100 \
    --min_tables 10 \
    --max_retr 10000
