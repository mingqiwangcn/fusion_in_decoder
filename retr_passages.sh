python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/nq_tables_index/populated.index \
    --meta_file ./data/nq_tables_index/passage_meta_all.jsonl \
    --passage_file ~/code/open_table_discovery/table2txt/dataset/nq_tables/triple_template_graph/graph_passages.json \
    --data ~/code/open_table_discovery/fusion_query_dev.jsonl \
    --output_path ./data/retrieved_dev.json \
    --n-docs 500 \
