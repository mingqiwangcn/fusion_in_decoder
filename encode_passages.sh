part=$1
python generate_passage_embeddings.py \
        --model_path ./pretrained_models/tqa_retriever \
        --passages ./data/passage_part_${part}.tsv \
        --output_path ./data/nq_tables_passage_embeddings_part_${part} \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 1000 \
