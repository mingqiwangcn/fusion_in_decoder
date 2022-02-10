if [ "$#" -ne 4 ]; then
  echo "Usage: ./retr_passages_syt.sh <dataset> <mode> <experiment> <passage_file>"
  exit
fi

dataset=$1
mode=$2
expr=$3
file_name=$4
sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/sql_all_per_10
out_dir=${sql_data_dir}/${expr}

python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ./data/on_disk_index_${dataset}_${expr}/${file_name} \
    --data ${sql_data_dir}/fusion_query_${mode}.jsonl \
    --output_path ${out_dir}/fusion_retrieved_${mode}.jsonl \
    --n-docs 100 \
