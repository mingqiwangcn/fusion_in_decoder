if [ "$#" -ne 3 ]; then
  echo "Usage: ./retr_passages_syt.sh <dataset> <experiment> <mode>"
  exit
fi

dataset=$1
expr=$2
mode=$3
file_name=token_text.jsonl
sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/auto_sql
out_dir=${sql_data_dir}/${expr}

python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ./data/on_disk_index_${dataset}_${expr}/${file_name} \
    --data ${sql_data_dir}/fusion_query_${mode}.jsonl \
    --output_path ${out_dir}/fusion_retrieved_${mode}.jsonl \
    --n-docs 100 \
