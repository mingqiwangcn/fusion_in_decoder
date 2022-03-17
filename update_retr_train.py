import json
from tqdm import tqdm
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--expr', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    data_dir = '/home/cc/code/open_table_discovery/table2txt/dataset/' 
    input_file = os.path.join(data_dir, args.dataset, args.expr, 'fusion_retrieved_train.jsonl_bak')
    out_file = os.path.join(data_dir, args.dataset, args.expr, 'fusion_retrieved_train.jsonl.updated')
    f_o = open(out_file, 'w')

    top_n = 100
    with open(input_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            gold_table_lst = item['table_id_lst']
            ctxs = item['ctxs'][:top_n]
            labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctxs]
            if (max(labels) < 1) or (min(labels) > 0):
                continue
            item['ctxs'] = ctxs
            f_o.write(json.dumps(item) + '\n')
    f_o.close()
    
    print('output to [%s]' % out_file)

if __name__ == '__main__':
    main()
