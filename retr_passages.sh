python passage_retrieval.py \
    --model_path ./pretrained_models/tqa_retriever \
    --passages "./data/nq_tables_passages/passage_small.tsv" \
    --data ./data/fusion_in_decoder_data.jsonl \
    --passages_embeddings "./data/nq_tables_passage_embeddings/nq_tables_passage_embeddings_small" \
    --output_path ./data/retrieved_data.json \
    --n-docs 100 \
