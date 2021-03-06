if [ "$#" -ne 5 ]; then
  echo "Usage: ./retr_passages_syt.sh <dataset> <experiment> <passage file> <sql_expr> <part>"
  exit
fi

dataset=$1
expr=$2
file_name=$3
sql_expr=$4
part=$5
sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}
out_dir=${sql_data_dir}/${expr}

python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ./data/on_disk_index_${dataset}_${expr}/${file_name} \
    --data ${sql_data_dir}/fusion_query.jsonl_${part} \
    --output_path ${out_dir}/fusion_retrieved.jsonl_${part} \
    --n-docs 100 \
    --min_tables 10 \
    --max_retr 10000
