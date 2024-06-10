import os
import torch
import numpy as np
from collections import OrderedDict
import random
from utils.utils import get_metadata
import torch.nn.functional as f
from torch.utils.data import Dataset
from tqdm import tqdm


class VCDataset(Dataset):

    def __init__(self, meta_path, hparams=None):
        '''
        :param meta_path: file list
        :param hparams: config file
        '''
        self.hparams = hparams
        self.abs_path = os.path.abspath(os.path.dirname(meta_path))
        self.file_list = get_metadata(meta_path+ '/utt_filter_1k8.lst')

            
        self.feature_menu = ['first_seqs', 'full_seqs', 'seq_lens', 'pos_ids', 'seq_tags']
        #with one 0 padding
        self.eos_id = self.hparams.num_semantic + self.hparams.num_acoustic + 1
        # self.phone_map = {}

                
                
            


    def _segment_crop(self, feature, max, min=0):
        '''
        crop length for VC since the limited capacity of GPU
        crop length between (0,min:max)
        :return: cropped feature, corp index
        '''
        assert max >= min
        crop_length = np.random.random_integers(min,max)
        feature = feature[:crop_length]
        return feature,crop_length


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        '''
        :param item: index in file list
        :return: semantic token, acoustic token
        '''
        acoustic_path, semantic_path = self.file_list[item].split('|')
        
        # load acoustic id
        acoustic_token = np.load(acoustic_path) if acoustic_path[0] == '/' else np.load(os.path.join(self.abs_path, acoustic_path))
        semantic_token = np.load(semantic_path) if semantic_path[0] == '/' else np.load(os.path.join(self.abs_path, semantic_path))
        #crop for VC
        acoustic_token, crop_length = self._segment_crop(acoustic_token, max = self.hparams.max_crop_length, min = self.hparams.min_crop_length)
        semantic_token = semantic_token[:int(crop_length*self.hparams.length_ratio)]
        #####
        # process tokens to input shape
        #####

        #token offset
        semantic_token = semantic_token + 1  # padding
        acoustic_token = acoustic_token.T + 1 + self.hparams.num_semantic  # padding + len(text)   640 for hubert
        # 
        
        # get len
        semantic_len = semantic_token.shape[0]
        acoustic_len, dim_size = acoustic_token.shape
        dtype = acoustic_token.dtype
        # concat text & wav, add EOS
        first_seq = np.asarray(list(semantic_token) + [self.eos_id] + list(acoustic_token[:, 0]) + [self.eos_id + 1])

        # concat text & wav, add EOS, full codebook
        full_semantic_seq = np.stack([semantic_token] * dim_size, axis=1)  # [t_semantic, dim size]
        eos1 = np.stack([np.asarray([self.eos_id, ])] * dim_size, axis=1)  # [1, dim size]
        full_seq = np.concatenate([full_semantic_seq, eos1, acoustic_token, eos1 + 1], axis=0)  # [t, n_codebook]

        # get position embedding
        pos_id = np.asarray(list(range(semantic_len + 1)) + list(range(acoustic_len + 1)))
        seq_tag = np.asarray([1] * (semantic_len + 1) + [2] * (acoustic_len + 1))  # mark the different input

        return first_seq, full_seq, pos_id, seq_tag

    def collate_fn(self, batch):
        first_seqs = []
        full_seqs = []
        pos_ids = []
        seq_tags = []
        seq_lens = []
        max_len = max(seq.shape[0] for seq, _, _, _ in batch)
        for first_seq, full_seq, pos_id, seq_tag in batch:
            seq_lens.append(first_seq.shape[0])
            pos_id = np.pad(pos_id, (0, max_len - first_seq.shape[0]), mode='constant', constant_values=0)
            seq_tag = np.pad(seq_tag,
                                (0, max_len - first_seq.shape[0]),
                                mode='constant',
                                constant_values=0)
            first_seq = np.pad(first_seq,
                         (0, max_len - first_seq.shape[0]),
                         mode='constant',
                         constant_values=0)
            full_seq = np.pad(full_seq,
                              [(0, max_len - full_seq.shape[0]), (0, 0)],
                              mode='constant',
                              constant_values=0)  # [t, n_codebook]
            first_seqs.append(first_seq)
            full_seqs.append(full_seq)
            pos_ids.append(pos_id)
            seq_tags.append(seq_tag)

        # to torch tensor
        first_seqs = torch.from_numpy(np.asarray(first_seqs))
        seq_lens = torch.from_numpy(np.asarray(seq_lens))
        full_seqs = torch.from_numpy(np.asarray(full_seqs))
        pos_ids = torch.from_numpy(np.asarray(pos_ids))
        seq_tags = torch.from_numpy(np.asarray(seq_tags))
        return first_seqs, full_seqs, seq_lens, pos_ids, seq_tags





class TTSDataset(Dataset):

    def __init__(self, meta_path, hparams=None):
        '''
        :param meta_path: file list
        :param hparams: config file
        '''
        self.hparams = hparams
        self.abs_path = os.path.abspath(meta_path)
        
        self.file_list = get_metadata(meta_path+ '/wenet_norm_4.lst')

   
        self.feature_menu = ['first_seqs', 'full_seqs','seq_lens', 'pos_ids', 'seq_tags']
        # with one 0 padding
        self.eos_id = self.hparams.num_semantic + self.hparams.num_acoustic + 1
        # self.phone_map = {}
        # tmp_lst=[]
        # for utt in tqdm(self.file_list):
            
            # try:
            #     acoustic_path, semantic_path = utt.split('|')
            #     acoustic_token = np.load(acoustic_path) if acoustic_path[0] == '/' else np.load(os.path.join(self.abs_path, acoustic_path))
            #     semantic_token = np.load(semantic_path) if semantic_path[0] == '/' else np.load(os.path.join(self.abs_path, semantic_path))
            #     semantic_token = semantic_token + 1  # padding
            #     acoustic_token = acoustic_token.T + 1 + self.hparams.num_semantic
            #     first_seq = np.asarray(list(semantic_token) + [self.eos_id] + list(acoustic_token[:, 0]) + [self.eos_id + 1])
            #     if first_seq.shape[0] > 0 and first_seq.shape[0] < 2048:
            #         tmp_lst.append(utt)
            #     # else:
            #     #     print(utt)
            # except:
            #     print(utt)
            
        # with open(meta_path+ '/utt_filter_1k8_1110.lst','w') as f:
        #     f.write('\n'.join(tmp_lst))    
        # self.file_list=tmp_lst       
            


    # def _segment_crop(self, feature, max, min=0):
    #     '''
    #     crop length for VC since the limited capacity of GPU
    #     crop length between (0,min:max)
    #     :return: cropped feature, corp index
    #     '''
    #     assert max >= min
    #     crop_length = np.random.random_integers(min,max)
    #     feature = feature[:crop_length]
    #     return feature,crop_length


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        '''
        :param item: index in file list
        :return: semantic token, acoustic token
        '''
        acoustic_path, semantic_path = self.file_list[item].split('|')
        
        # load acoustic id
        acoustic_token = np.load(acoustic_path) if acoustic_path[0] == '/' else np.load(os.path.join(self.abs_path, acoustic_path))
        semantic_token = np.load(semantic_path) if semantic_path[0] == '/' else np.load(os.path.join(self.abs_path, semantic_path))
        #crop for VC
        # acoustic_token, crop_length = self._segment_crop(acoustic_token, max = self.hparams.max_crop_length, min = self.hparams.min_crop_length)
        # semantic_token = semantic_token[:int(crop_length*self.hparams.length_ratio)]
        #####
        # process tokens to input shape
        #####

        #token offset
        semantic_token = semantic_token + 1  # padding
        acoustic_token = acoustic_token.T + 1 + self.hparams.num_semantic  # padding + len(text)   640 for hubert
        # 
        
        # get len
        semantic_len = semantic_token.shape[0]
        acoustic_len, dim_size = acoustic_token.shape
        dtype = acoustic_token.dtype
        # concat text & wav, add EOS
        first_seq = np.asarray(list(semantic_token) + [self.eos_id] + list(acoustic_token[:, 0]) + [self.eos_id + 1])

        # concat text & wav, add EOS, full codebook
        full_semantic_seq = np.stack([semantic_token] * dim_size, axis=1)  # [t_semantic, dim size]
        eos1 = np.stack([np.asarray([self.eos_id, ])] * dim_size, axis=1)  # [1, dim size]
        full_seq = np.concatenate([full_semantic_seq, eos1, acoustic_token, eos1 + 1], axis=0)  # [t, n_codebook]

        # get position embedding
        pos_id = np.asarray(list(range(semantic_len + 1)) + list(range(acoustic_len + 1)))
        seq_tag = np.asarray([1] * (semantic_len + 1) + [2] * (acoustic_len + 1))  # mark the different input

        return first_seq, full_seq, pos_id, seq_tag

    def collate_fn(self, batch):
        first_seqs = []
        full_seqs = []
        pos_ids = []
        seq_tags = []
        seq_lens = []
        max_len = max(seq.shape[0] for seq, _, _, _ in batch)
        for first_seq, full_seq, pos_id, seq_tag in batch:
            seq_lens.append(first_seq.shape[0])
            pos_id = np.pad(pos_id, (0, max_len - first_seq.shape[0]), mode='constant', constant_values=0)
            seq_tag = np.pad(seq_tag,
                                (0, max_len - first_seq.shape[0]),
                                mode='constant',
                                constant_values=0)
            first_seq = np.pad(first_seq,
                         (0, max_len - first_seq.shape[0]),
                         mode='constant',
                         constant_values=0)
            full_seq = np.pad(full_seq,
                              [(0, max_len - full_seq.shape[0]), (0, 0)],
                              mode='constant',
                              constant_values=0)  # [t, n_codebook]
            first_seqs.append(first_seq)
            full_seqs.append(full_seq)
            pos_ids.append(pos_id)
            seq_tags.append(seq_tag)

        # to torch tensor
        first_seqs = torch.from_numpy(np.asarray(first_seqs))
        seq_lens = torch.from_numpy(np.asarray(seq_lens))
        full_seqs = torch.from_numpy(np.asarray(full_seqs))
        pos_ids = torch.from_numpy(np.asarray(pos_ids))
        seq_tags = torch.from_numpy(np.asarray(seq_tags))
        return first_seqs, full_seqs, seq_lens, pos_ids, seq_tags


class WNDataset(Dataset):

    def __init__(self, meta_path, hparams=None):
        '''
        :param meta_path: file list
        :param hparams: config file
        '''
        self.hparams = hparams
        self.abs_path = os.path.abspath(meta_path)
        
        self.file_list = get_metadata(meta_path+ '/wenet_new.lst')

   
        self.feature_menu = ['first_seqs', 'seq_lens', 'pos_ids', 'seq_tags']
        # with one 0 padding
        self.eos_id = self.hparams.num_semantic + self.hparams.num_acoustic + 1
        # self.phone_map = {}
        # tmp_lst=[]
        # for utt in tqdm(self.file_list):
            
            # try:
            #     acoustic_path, semantic_path = utt.split('|')
            #     acoustic_token = np.load(acoustic_path) if acoustic_path[0] == '/' else np.load(os.path.join(self.abs_path, acoustic_path))
            #     semantic_token = np.load(semantic_path) if semantic_path[0] == '/' else np.load(os.path.join(self.abs_path, semantic_path))
            #     semantic_token = semantic_token + 1  # padding
            #     acoustic_token = acoustic_token.T + 1 + self.hparams.num_semantic
            #     first_seq = np.asarray(list(semantic_token) + [self.eos_id] + list(acoustic_token[:, 0]) + [self.eos_id + 1])
            #     if first_seq.shape[0] > 0 and first_seq.shape[0] < 2048:
            #         tmp_lst.append(utt)
            #     # else:
            #     #     print(utt)
            # except:
            #     print(utt)
            
        # with open(meta_path+ '/utt_filter_1k8_1110.lst','w') as f:
        #     f.write('\n'.join(tmp_lst))    
        # self.file_list=tmp_lst       
            


    # def _segment_crop(self, feature, max, min=0):
    #     '''
    #     crop length for VC since the limited capacity of GPU
    #     crop length between (0,min:max)
    #     :return: cropped feature, corp index
    #     '''
    #     assert max >= min
    #     crop_length = np.random.random_integers(min,max)
    #     feature = feature[:crop_length]
    #     return feature,crop_length


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        '''
        :param item: index in file list
        :return: semantic token, acoustic token
        '''
        acoustic_path, semantic_path = self.file_list[item].split('|')
        
        # load acoustic id
        acoustic_token = np.load(acoustic_path) if acoustic_path[0] == '/' else np.load(os.path.join(self.abs_path, acoustic_path))
        semantic_token = np.load(semantic_path) if semantic_path[0] == '/' else np.load(os.path.join(self.abs_path, semantic_path))
        #crop for VC
        # acoustic_token, crop_length = self._segment_crop(acoustic_token, max = self.hparams.max_crop_length, min = self.hparams.min_crop_length)
        # semantic_token = semantic_token[:int(crop_length*self.hparams.length_ratio)]
        #####
        # process tokens to input shape
        #####

        #token offset
        semantic_token = semantic_token + 1  # padding
        acoustic_token = acoustic_token.T + 1 + self.hparams.num_semantic  # padding + len(text)   640 for hubert
        # 
        
        # get len
        semantic_len = semantic_token.shape[0]
        acoustic_len, dim_size = acoustic_token.shape
        dtype = acoustic_token.dtype
        # concat text & wav, add EOS
        first_seq = np.asarray(list(semantic_token) + [self.eos_id] + list(acoustic_token[:, 0]) + [self.eos_id + 1])

        # concat text & wav, add EOS, full codebook
        full_semantic_seq = np.stack([semantic_token] * dim_size, axis=1)  # [t_semantic, dim size]
        eos1 = np.stack([np.asarray([self.eos_id, ])] * dim_size, axis=1)  # [1, dim size]
        full_seq = np.concatenate([full_semantic_seq, eos1, acoustic_token, eos1 + 1], axis=0)  # [t, n_codebook]

        # get position embedding
        pos_id = np.asarray(list(range(semantic_len + 1)) + list(range(acoustic_len + 1)))
        seq_tag = np.asarray([1] * (semantic_len + 1) + [2] * (acoustic_len + 1))  # mark the different input

        return first_seq, full_seq, pos_id, seq_tag

    def collate_fn(self, batch):
        first_seqs = []
        full_seqs = []
        pos_ids = []
        seq_tags = []
        seq_lens = []
        max_len = max(seq.shape[0] for seq, _, _, _ in batch)
        for first_seq, full_seq, pos_id, seq_tag in batch:
            seq_lens.append(first_seq.shape[0])
            pos_id = np.pad(pos_id, (0, max_len - first_seq.shape[0]), mode='constant', constant_values=0)
            seq_tag = np.pad(seq_tag,
                                (0, max_len - first_seq.shape[0]),
                                mode='constant',
                                constant_values=0)
            first_seq = np.pad(first_seq,
                         (0, max_len - first_seq.shape[0]),
                         mode='constant',
                         constant_values=0)
            full_seq = np.pad(full_seq,
                              [(0, max_len - full_seq.shape[0]), (0, 0)],
                              mode='constant',
                              constant_values=0)  # [t, n_codebook]
            first_seqs.append(first_seq)
            full_seqs.append(full_seq)
            pos_ids.append(pos_id)
            seq_tags.append(seq_tag)

        # to torch tensor
        first_seqs = torch.from_numpy(np.asarray(first_seqs))
        seq_lens = torch.from_numpy(np.asarray(seq_lens))
        full_seqs = torch.from_numpy(np.asarray(full_seqs))
        pos_ids = torch.from_numpy(np.asarray(pos_ids))
        seq_tags = torch.from_numpy(np.asarray(seq_tags))
        return first_seqs, full_seqs, seq_lens, pos_ids, seq_tags


if __name__ == "__main__":
    train_datatset = VCDataset('test', hparams=None)


