python passage_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --passages "./data/nq_tables_passages/passage_part_*.tsv" \
    --data ~/code/open_table_discovery/dataset/nq_tables/triple_template_graph/dev/fusion_in_decoder_data.jsonl \
    --passages_embeddings "./data/nq_tables_passage_embeddings/nq_tables_passage_embeddings_part_*_00" \
    --output_path ./data/retrieved_data.json \
    --n-docs 100 \
