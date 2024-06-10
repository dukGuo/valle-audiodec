import os
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
import random
from utils.utils import get_metadata
import torch.nn.functional as f
np.set_printoptions(threshold=np.inf)


class VCDataset(Dataset):

    def __init__(self, meta_path, hparams):
        self.meta_path = meta_path
        self.abs_path = os.path.abspath(os.path.dirname(meta_path))
        self.hparams = hparams
        self.metas = get_metadata(meta_path)
        self.eos_id = self.hparams.num_semantic + self.hparams.num_acoustic + 1
        self.feature_menu = ['first_seqs', 'full_seqs', 'seq_lens', 'pos_ids', 'seq_tags','utts']

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, item):
        acoustic_path, semantic_path = self.metas[item].split('|')
        # load wav id
        acoustic_token = np.load(acoustic_path) if acoustic_path[0] == '/' else np.load(
            os.path.join(self.abs_path, acoustic_path))
        semantic_token = np.load(semantic_path) if semantic_path[0] == '/' else np.load(
            os.path.join(self.abs_path, semantic_path))
        utt = os.path.basename(acoustic_path)

        t_size, dim_size = semantic_token.shape

        semantic_token = semantic_token + 1  # padding
        acoustic_token = acoustic_token + 1 + self.hparams.num_acoustic  # padding + len(text)

        # get len
        acoustic_len = acoustic_token.shape[0]
        semantic_len = semantic_token.shape[0]

        # concat text & wav, add EOS
        first_seq = list(semantic_token) + [self.eos_id] + list(acoustic_token[:, 0])
        first_seq = np.asarray(first_seq)


        full_semantic_seq = np.stack([semantic_token] * dim_size, axis=1)  # [t,] -> [t, n_codebook]
        eos1 = np.stack([np.asarray([self.eos_id, ])] * dim_size, axis=1)  # [1, n_codebook]
        # full_seq = np.concatenate([full_text_seq, eos1, wav_id, eos1 + 1], axis=0) # [t, n_codebook]
        full_seq = np.concatenate([full_semantic_seq, eos1, acoustic_token], axis=0)  # [t, n_codebook]

        pos_id = np.asarray(list(range(semantic_len + 1)) + list(range(acoustic_len)))
        seq_tag = np.asarray([1] * (semantic_len + 1) + [2] * (acoustic_len))

        return first_seq, full_seq, pos_id, seq_tag, utt[:-4]

    def collate_fn(self, batches):
        # length padding
        first_seqs = []
        seq_lens = []
        seq_tags = []
        pos_ids = []
        full_seqs = []
        utts=[]

        max_seq_len = max(first_seq.shape[0] for first_seq, _, _, _,_,_,_ in batches)
        for first_seq, pos_id, full_seq, seq_sen_id, utt in batches:
            seq_lens.append(first_seq.shape[0])
            # position id
            pos_id = np.pad(pos_id, (0, max_seq_len - first_seq.shape[0]), mode='constant', constant_values=0)
            # 区分是text还是wav
            seq_sen_id = np.pad(seq_sen_id,
                                (0, max_seq_len - first_seq.shape[0]),
                                mode='constant',
                                constant_values=0)
            first_seq = np.pad(first_seq,
                         (0, max_seq_len - first_seq.shape[0]),
                         mode='constant',
                         constant_values=0)
            full_seq = np.pad(full_seq,
                              [(0, max_seq_len - full_seq.shape[0]), (0, 0)],
                              mode='constant',
                              constant_values=0) # [t, n_codebook]
            first_seqs.append(first_seq)
            full_seqs.append(full_seq)
            pos_ids.append(pos_id)
            seq_tags.append(seq_sen_id)
            utts.append(utt)
        # to torch
        first_seqs = torch.from_numpy(np.asarray(first_seqs))
        seq_lens = torch.from_numpy(np.asarray(seq_lens))
        full_seqs = torch.from_numpy(np.asarray(full_seqs))
        pos_ids = torch.from_numpy(np.asarray(pos_ids))
        seq_tags = torch.from_numpy(np.asarray(seq_tags))
        return first_seqs, full_seqs, seq_lens, pos_ids, seq_tags, utts
