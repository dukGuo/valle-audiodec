import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn.functional as F
from tqdm import tqdm


class AR(nn.Module):
    def __init__(self, hparams, device):
        super().__init__()
        self.hparams = hparams
        self.device = device
        self.lm_model = GPT2LMHeadModel(config=self._model_config()).to(device)

    def _model_config(self):
        return GPT2Config(
            vocab_size=self.hparams.GPT2_vocab_size,  
            n_positions=self.hparams.GPT2_n_positions, 
            n_ctx=self.hparams.GPT2_n_ctx,  
            n_embd=self.hparams.GPT2_n_embd,  
            n_layer=self.hparams.GPT2_n_layer,  
            n_head=self.hparams.GPT2_n_head,  
        )


    def inference(self, data,max_len=None,topk=None):
        idx = data['first_idx']
        pos_ids = data['pos_ids']
        tags = data['tags']
        semantic_len = (tags == 1).sum(dim=1)
        prompt_acoustic_len = (tags == 2).sum(dim=1)
        data['prompt_len'] = prompt_acoustic_len
        
        if max_len == None:
            max_len = int((semantic_len - 1) * 320 / 16000 * 24000 / 300 - prompt_acoustic_len) + 3

        for j in tqdm(range(max_len)):
            lm_outputs = self.lm_model(
                input_ids=idx,
                attention_mask=None,
                position_ids=pos_ids,
            )
            logits = lm_outputs['logits']
            logits = logits * self.hparams.temperature
            logits[:, :, 0:(self.hparams.num_semantic+1)] = -float('Inf') # semantic token 
            logits[:, :, self.hparams.num_semantic + 1 + self.hparams.num_acoustic] =-float('Inf') # eos
            
            logits=logits[:, -1, :]
            if topk is not None:
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            
            probs = logits.softmax(dim=-1)  # [b, d]
            dist = torch.distributions.categorical.Categorical(probs=probs)
            samples = dist.sample().unsqueeze(0)
            
            idx = torch.cat([idx, samples], dim=1)  # [b, t]
            pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
            tags = torch.cat([tags, torch.zeros_like(tags[:, -1:]) + 2], dim=1)
            if max_len is not None:
                if samples.item() == self.hparams.num_semantic + 1 + self.hparams.num_acoustic +1: # eos
                    break
                if j == max_len-1:
                    # too long break
                    samples[:,:] = self.hparams.num_semantic + 1 + self.hparams.num_acoustic +1
                    idx = torch.cat([idx, samples], dim=1) 
                    pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                    tags = torch.cat([tags, torch.zeros_like(tags[:, -1:]) + 2], dim=1)
                    break
        data['tags'] = tags
        data['pos_ids'] = pos_ids
        data['first_idx'] = idx
        return idx
