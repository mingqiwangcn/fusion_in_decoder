python ./passage_ondisk_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --index_file ./data/nq_tables_index/populated.index \
    --meta_file ./data/nq_tables_index/passage_meta_all.jsonl \
    --passage_file ./data/nq_tables_index/graph_passages.json \
    --data ./data/fusion_query_test.jsonl \
    --output_path ./data/retrieved_test.json \
    --n-docs 500 \
