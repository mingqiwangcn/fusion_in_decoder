if [ "$#" -ne 4 ]; then
    echo "Usage: ./index_passages.sh <work_dir> <dataset> <experiment> <emb_file>"
    exit
fi

work_dir=$1
dataset=$2
exptr=$3
emb_file=$4

python ./src/ondisk_index.py \
    --work_dir ${work_dir} \
    --dataset ${dataset} \
    --experiment ${exptr} \
    --emb_file ${emb_file} \
