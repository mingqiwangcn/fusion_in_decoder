mode=$1
dataset=nq_tables
expr=rel_graph
file_name=passages.jsonl
sql_expr=sql_data
sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}

python ./passage_ondisk_retrieval.py \
    --model_path ~/code/models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ./data/on_disk_index_${dataset}_${expr}/${file_name} \
    --data ${sql_data_dir}/${mode}/fusion_query.jsonl \
    --output_path ${sql_data_dir}/${mode}/${expr}/fusion_retrieved.jsonl \
    --n-docs 100 \
    --min_tables 5 \
    --max_retr 1000
