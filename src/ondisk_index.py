import faiss
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm
import pickle
import random
import os
import numpy as np
import uuid

class OndiskIndexer:
    @staticmethod
    def index_data(index_file, data_file, block_size=5000000):
        print('loading data')
        with open(data_file, 'rb') as f:
            p_ids, p_embs = pickle.load(f)
       
        tmp_dir = 'ondisk_index_tmp_%s' % str(uuid.uuid4())
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

    @staticmethod
    def create(data_file, index_file):
        if os.path.exists(index_file):
            print('index file [%s] already exists' % index_file)
            return 
        print('loading data')
        with open(data_file, 'rb') as f:
            p_ids, p_embs = pickle.load(f)
        
        N = p_embs.shape[0]
        D = p_embs.shape[1] 
        index = faiss.index_factory(D, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
        
        train_rows = random.sample(list(np.arange(0, N)), N // 10) 
        train_embs = np.vstack([p_embs[a] for a in train_rows])
        train_embs = np.float32(train_embs)
        print('training index')
        index.train(train_embs)
        print('wrting index to [%s]' % index_file)
        faiss.write_index(index, index_file) 

def create_index():
    data_file = '../data/nq_tables_passage_embeddings_sorted/nq_tables_passage_embeddings_all'
    index_file = '../data/nq_tables.index'
    #OndiskIndexer.create(data_file, index_file)
    #OndiskIndexer.index_data(index_file, data_file)

def test_query():
    index_file = './nq_tables_index/populated.index' 
    index = faiss.read_index(index_file)
    index.nprobe = 16
    
    d = 768
    qr = faiss.randn((2, d), 123)
    dists, ids = index.search(qr, 6)
    print()

def main():
    test_query()

if __name__ == '__main__':
    main()
    
     
