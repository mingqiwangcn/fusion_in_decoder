import json
from tqdm import tqdm
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()
    return args

def get_top_passages(item):
    num_max_passages = 10
    ctx_lst = item['ctxs']
    updated_ctx_lst = []
    table_info_dict = {}
    for ctx in ctx_lst:
        table_id = ctx['tag']['table_id']
        num_passages = table_info_dict.get(table_id, 0)
        if num_passages < num_max_passages:
            updated_ctx_lst.append(ctx)
            table_info_dict[table_id] = num_passages + 1 
    item['ctxs'] = updated_ctx_lst

def main():
    args = get_args()
    input_file = os.path.join(args.data_dir, 'fusion_retrieved_%s.jsonl_bak' % args.mode) 
    out_file = os.path.join(args.data_dir, 'fusion_retrieved_%s.jsonl' % args.mode)
    if os.path.exists(out_file):
        print('output file (%s) already exists' % out_file)
        return

    is_train = (args.mode == 'train')
    f_o = open(out_file, 'w')

    top_n = 200
    with open(input_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            get_top_passages(item)
            gold_table_lst = item['table_id_lst']
            ctxs = item['ctxs'][:top_n]
            
            if is_train:
                labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctxs]
                if (max(labels) < 1) or (min(labels) > 0):
                    continue
            
            item['ctxs'] = ctxs
            f_o.write(json.dumps(item) + '\n')
    f_o.close()
    
    print('output to [%s]' % out_file)

if __name__ == '__main__':
    main()
