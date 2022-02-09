if [ "$#" -ne 3 ]; then
  echo "Usage: ./retr_passages_syt.sh <dataset> <experiment> <passage_file>"
  exit
fi

dataset=$1
expr=$2
passage_file=$3
sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/sql_all_per_10
out_dir=${sql_data_dir}/${expr}

python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ~/code/open_table_discovery/table2txt/dataset/${dataset}/${expr}/${passage_file} \
    --data ${sql_data_dir}/fusion_query.jsonl \
    --output_path ${out_dir}/retrieved_data.jsonl \
    --n-docs 100 \
