if [ "$#" -ne 2 ]; then
  echo "Usage: ./retr_passages_syt.sh <dataset> <experiment>"
  exit
fi

dataset=$1
expr=$2
file_name=passages.jsonl
sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/auto_sql
out_dir=${sql_data_dir}/${expr}

python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ./data/on_disk_index_${dataset}_${expr}/${file_name} \
    --data ${sql_data_dir}/fusion_query.jsonl \
    --output_path ${out_dir}/fusion_retrieved.jsonl \
    --n-docs 500 \
