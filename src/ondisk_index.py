import faiss
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm
import pickle
import random
import os
import numpy as np
import uuid
import json
import argparse

class OndiskIndexer:
    def __init__(self, index_file, meta_file, passage_file):
        self.index = faiss.read_index(index_file)
        self.meta_dict = self.load_passage_meta(meta_file, passage_file)
   
    def get_passage_key(self, tag):
        table_id = tag['table_id']
        row = tag['row']
        sub_col = tag['sub_col']
        obj_col = tag['obj_col']
        key = '%s_%s_%s_%s' % (table_id, str(row), str(sub_col), str(obj_col))
        return key 

    def load_passage_meta(self, meta_file, passage_file):
        passage_dict = {} 
        with open(passage_file) as f_p:
            for line in tqdm(f_p):
                item = json.loads(line)
                tag = item['tag']
                key = self.get_passage_key(tag)
                if key in passage_dict:
                    raise ValueError('duplicate passage keys')
                passage_dict[key] = item['passage']         

        meta_dict = {}
        with open(meta_file) as f_m:
            for line in tqdm(f_m):
                item = json.loads(line)
                p_id = item['p_id']
                tag = item['tag']
                key = self.get_passage_key(tag)
                passage = passage_dict[key]
                item['passage'] = passage
                meta_dict[str(p_id)] = item
        
        del passage_dict
        return meta_dict 
        
    
    def search(self, query, top_n=500, n_probe=16, batch_size=10):
        result = []
        N = len(query)
        for idx in tqdm(range(0, N, batch_size)):
            pos = idx + batch_size
            batch_query = query[idx:pos]
            batch_result = self.batch_search(batch_query, top_n=top_n, n_probe=n_probe)
            result.extend(batch_result)
        return result
         
    def batch_search(self, query, top_n=10, n_probe=16):
        self.index.nprobe = n_probe
        batch_dists, batch_p_ids = self.index.search(query, top_n)
        batch_result = []
        num_batch = len(query)
        for item_idx in range(num_batch):
            item_result = []
            p_id_lst = batch_p_ids[item_idx]
            p_dist_lst = batch_dists[item_idx]
            for idx, p_id in enumerate(p_id_lst):
                passage_info = self.meta_dict[str(p_id)]
                out_item = {
                    'p_id':p_id,
                    'passage':passage_info['passage'],
                    'score':p_dist_lst[idx],
                    'tag':passage_info['tag']
                }
                item_result.append(out_item)
            batch_result.append(item_result)
        return batch_result

    @staticmethod
    def index_data(index_file, data_file, block_size=5000000):
        print('loading data')
        with open(data_file, 'rb') as f:
            p_ids, p_embs = pickle.load(f)
       
        tmp_dir = 'ondisk_index_%s' % str(uuid.uuid4())
        os.mkdir(tmp_dir)
        N = len(p_ids)
        bno = 0
        block_fnames = []

        print('creating block indexes')
        for idx in range(0, N, block_size):
            index = faiss.read_index(index_file)
            index.set_direct_map_type(faiss.DirectMap.Hashtable)
            pos = idx + block_size
            block_p_ids = np.int64(np.array(p_ids[idx:pos]))
            block_p_embs = np.float32(p_embs[idx:pos])
            index.add_with_ids(block_p_embs, block_p_ids)
            block_file_name = os.path.join(tmp_dir, 'block_%d.index' % bno)
            faiss.write_index(index, block_file_name)
            block_fnames.append(block_file_name)
            bno += 1
       
        merged_file_name = os.path.join(tmp_dir, 'merged_index.ivfdata')
        print('merging block indexes')
        index = faiss.read_index(index_file)
        merge_ondisk(index, block_fnames, merged_file_name)
         
        out_index_file = os.path.join(tmp_dir, 'populated.index')
        print('writing to [%s]' % out_index_file)
        faiss.write_index(index, out_index_file)

    # create an empty index and train it
    @staticmethod
    def create(data_file, index_file, num_train):
        if os.path.exists(index_file):
            print('index file [%s] already exists' % index_file)
            return 
        print('loading data')
        with open(data_file, 'rb') as f:
            p_ids, p_embs = pickle.load(f)
        
        N = p_embs.shape[0]
        D = p_embs.shape[1] 
        index = faiss.index_factory(D, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
       
        all_rows = list(np.arange(0, N))
        if num_train > len(all_rows):
            raise ValueError('num_train must not be greater than size of data') 
        train_rows = random.sample(list(np.arange(0, N)), num_train) 
        train_embs = np.vstack([p_embs[a] for a in train_rows])
        train_embs = np.float32(train_embs)
        print('training index')
        index.train(train_embs)
        print('wrting index to [%s]' % index_file)
        faiss.write_index(index, index_file) 

def create_index(args):
    dataset_dir = '/home/cc/code/open_table_discovery/table2txt/dataset/'
    exptr_dir = os.path.join(dataset_dir, args.dataset, args.experiment)
    data_file = os.path.join(exptr_dir, args.emb_file)
    index_file = os.path.join(exptr_dir, '%s_%s.index' % (args.dataset, args.experiment))
    if os.path.exists(index_file):
        raise ValueError('Index [%s] already exists' % index_file)

    OndiskIndexer.create(data_file, index_file, args.num_train)
    OndiskIndexer.index_data(index_file, data_file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--emb_file', type=str)
    parser.add_argument('--num_train', type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    create_index(args)

if __name__ == '__main__':
    main()
    
     
