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
import glob
import math

class OndiskIndexer:
    def __init__(self, index_file, passage_file):
        self.index = faiss.read_index(index_file)
        self.passage_dict = self.load_passages(passage_file)
   
    def load_passages(self, passage_file):
        passage_dict = {} 
        with open(passage_file) as f_p:
            for line in tqdm(f_p):
                item = json.loads(line)
                p_id = item['p_id']
                passage_dict[int(p_id)] = item 
        return passage_dict
    
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
                passage_info = self.passage_dict[int(p_id)]
                out_item = {
                    'p_id':p_id,
                    'passage':passage_info['passage'],
                    'score':p_dist_lst[idx],
                    'tag':passage_info['tag']
                }
                item_result.append(out_item)
            batch_result.append(item_result)
        return batch_result

# end of class OndiskIndexer 

def index_data(index_file, data_file, index_out_dir, block_size=5000000):
    print('start indexing passages')
    bno = 0
    block_fnames = []
    
    emb_file_lst = glob.glob(data_file)
    emb_file_lst.sort()
    for emb_file in emb_file_lst: 
        print('loading file [%s]' % emb_file)
        with open(emb_file, 'rb') as f:
            p_ids, p_embs = pickle.load(f)
        N = len(p_ids)
        print('creating block indexes')
        for idx in range(0, N, block_size):
            index = faiss.read_index(index_file)
            index.set_direct_map_type(faiss.DirectMap.Hashtable)
            pos = idx + block_size
            block_p_ids = np.int64(np.array(p_ids[idx:pos]))
            block_p_embs = np.float32(p_embs[idx:pos])
            index.add_with_ids(block_p_embs, block_p_ids)
            block_file_name = os.path.join(index_out_dir, 'block_%d.index' % bno)
            faiss.write_index(index, block_file_name)
            block_fnames.append(block_file_name)
            bno += 1
   
    merged_file_name = os.path.join(index_out_dir, 'merged_index.ivfdata')
    print('merging block indexes')
    index = faiss.read_index(index_file)
    merge_ondisk(index, block_fnames, merged_file_name)
     
    out_index_file = os.path.join(index_out_dir, 'populated.index')
    print('writing to [%s]' % out_index_file)
    faiss.write_index(index, out_index_file)
   
    #remove the empty trained index and the block files
    os.remove(index_file)
    for block_file_name in block_fnames:
        os.remove(block_file_name)

def get_index_options(num_vecs):
    unit = 1e6 
    if num_vecs < unit:
        num_clusters = int(16 * math.sqrt(num_vecs))
        num_clusters = min(num_clusters, num_vecs)
        num_train = 256 * num_clusters
        num_train = min(num_train, num_vecs)
        factory_string = 'IVF%s,Flat' % num_clusters

    elif (num_vecs >= unit) and (num_vecs < 10 * unit):
       num_clusters = 65536
       num_train = min(60 * num_clusters, num_vecs)
       factory_string = 'IVF%s_HNSW32,Flat' % num_clusters
    
    elif (num_vecs >= 10 * unit) and (num_vecs < 100 * unit):
       num_clusters = 262144
       num_train = min(60 * num_clusters, num_vecs)
       factory_string = 'IVF%s_HNSW32,Flat' % num_clusters
        
    elif (num_vecs >= 100 * unit) and (num_vecs < 1000 * unit):
       num_clusters = 1048576
       num_train = min(60 * num_clusters, num_vecs)
       factory_string = 'IVF%s_HNSW32,Flat' % num_clusters
    
    else:
        raise ValueError('Not supported right now') 

    print('factory_string=%s, num_train=%d' % (factory_string, num_train))     
    return factory_string, num_train

def get_num_vecs(emb_file_lst):
    print('collecting the number of vectors')
    num_vecs = 0
    for emb_file in emb_file_lst:
        with open(emb_file, 'rb') as f:
            _, p_embs = pickle.load(f)
        num_part_vecs = len(p_embs)
        num_vecs += num_part_vecs
    return num_vecs

# create an empty index and train it
def create_train(data_file, index_file):
    if os.path.exists(index_file):
        print('index file [%s] already exists' % index_file)
        return 
    print('loading data')
    emb_file_lst = glob.glob(data_file)
    emb_file_lst.sort()

    num_vecs = get_num_vecs(emb_file_lst)
    print('num_vecs=%d' % num_vecs)
    factory_string, num_train = get_index_options(num_vecs)

    num_train_per_file = num_train // len(emb_file_lst)
    
    train_emb_lst = []
    for emb_file in emb_file_lst:
        with open(emb_file, 'rb') as f:
            _, p_embs = pickle.load(f)
        N = p_embs.shape[0]
        rows = list(np.arange(0, N))
        train_rows = random.sample(rows, num_train_per_file)
        train_emb = p_embs[train_rows] 
        train_emb_lst.append(train_emb)

    train_all_embs = np.vstack(train_emb_lst)
    train_all_embs = np.float32(train_all_embs)
   
    print('number of traing vectors = %d' % len(train_all_embs))
    
    D = train_all_embs.shape[1]
    index = faiss.index_factory(D, factory_string, faiss.METRIC_INNER_PRODUCT)
   
    print('training index')
    index.train(train_all_embs)
    print('wrting trained index to [%s]' % index_file)
    faiss.write_index(index, index_file) 

def main():
    args = get_args()
    if not os.path.isdir('data'):
        os.mkdir('data')

    index_out_dir = os.path.join('data', 'on_disk_index_%s_%s' % (args.dataset, args.experiment))
    if os.path.exists(index_out_dir):
        raise ValueError('Index directory [%s] already exists' % index_out_dir)
    
    os.mkdir(index_out_dir)
    dataset_dir = '/home/cc/code/open_table_discovery/table2txt/dataset/'
    exptr_dir = os.path.join(dataset_dir, args.dataset, args.experiment)
    data_file = os.path.join(exptr_dir, args.emb_file)
    trained_index_file = os.path.join(index_out_dir, 'trained.index')
    create_train(data_file, trained_index_file)
    index_data(trained_index_file, data_file, index_out_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--emb_file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        main()
    except ValueError as e:
        print(e)
    
    
     
