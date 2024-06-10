import  os
import torch
import numpy as np
from tqdm import tqdm
import time
import torchaudio
import soundfile as sf
import shutil

from models.ar import AR
from models.nar import  NAR
from utils.utils import *
from text.chinese import generate_token
from AudioDec.utils.audiodec import AudioDec, assign_model

def tokenize_wav(wav_path,audiodec,device,sample_rate=24000):
    
    wav, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)    
    with torch.no_grad():
        wav = wav.unsqueeze(1) #C T-> 1 C T  
        wav = wav.float().to(device)
        z = audiodec.tx_encoder.encode(wav)
        idx = audiodec.tx_encoder.quantize(z)
        
        inc = torch.arange(8)*1024
        idx = idx.cpu() - inc.reshape(-1,1)
        return idx.numpy().T
    

def do_tts(hp,args):
    
    
    # init codec
    model_name = "libritts_v1"
    device = args.device 
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
    audiodec = AudioDec(tx_device=device , rx_device=device )
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
    
    # init valle
    ar = AR(hp,device)
    ar_ckpt = torch.load(args.ar_ckpt, map_location=device)
    ar.load_state_dict(ar_ckpt['model'])
    ar.eval()
    nar = NAR(hp, device).to(device)
    nar_ckpt = torch.load(args.nar_ckpt, map_location=device)
    nar_weight = nar_ckpt["model"]
    if list(nar_weight.keys())[0].startswith('_orig'):
        nar_weight = {}
        for k, v in nar_ckpt["model"].items():
            nar_weight[k.split("_orig_mod.")[1]] = v
    nar.load_state_dict(nar_weight)
    nar.eval()
    
    # prepare data
    prompt_text = np.array(generate_token(args.prompt_text))
    prompt_token = tokenize_wav(args.prompt_wav,audiodec,device,sample_rate)
    text = np.array(generate_token(args.text))
    
    semantic_token = torch.from_numpy(np.concatenate((prompt_text,text)))
    t_size, dim_size = prompt_token.shape
    semantic_token = semantic_token + 1   # padding
    acoustic_token = prompt_token + 1 + hp.num_semantic # padding + len(text)
    
    acoustic_len = prompt_token.shape[0]
    semantic_len = semantic_token.shape[0]
    max_len = 2040 - semantic_len-acoustic_len
    eos_id = hp.num_semantic + hp.num_acoustic + 1
    
    first_idx = np.asarray(list(semantic_token) + [eos_id] + list(acoustic_token[:, 0]))
    full_semantic = np.stack([semantic_token] * dim_size, axis=1)  # t,8
    eos_full = np.stack([np.asarray([eos_id, ])] * dim_size, axis=1)  # 1, 8
    full_idx = np.concatenate([full_semantic, eos_full, acoustic_token], axis=0)  
        

    pos_ids= np.asarray(list(range(semantic_len + 1)) + list(range(acoustic_len)))
    tags = np.asarray([1] * (semantic_len + 1) + [2] * (acoustic_len))            
           
    data = {}    
    data["first_idx"] = torch.from_numpy(np.asarray(first_idx)).unsqueeze(0).to(device)
    data["seq_lens"] = torch.from_numpy(np.asarray(first_idx.shape[0])).unsqueeze(0).to(device)
    data["full_idx"] = torch.from_numpy(np.asarray(full_idx)).unsqueeze(0).to(device)
    data["pos_ids"] = torch.from_numpy(np.asarray(pos_ids)).unsqueeze(0).to(device)
    data["tags"] = torch.from_numpy(np.asarray(tags)).unsqueeze(0).to(device)
    
    
    # infer
    with torch.no_grad():
        first_idx = ar.inference(data,max_len,20)
        full_idx = nar.inference(data)        
        full_idx = (full_idx - (hp.num_semantic+1))[:, :, :-1] 
             
        prompt_idx = full_idx[:,:,:data['prompt_len']]        
        full_idx = full_idx[:,:,data['prompt_len']:]
        full_idx = full_idx.squeeze(0)
        
        inc = (torch.arange(8)*1024).to(device)
        full_idx =  full_idx + inc.reshape(-1,1)
        zq = audiodec.rx_encoder.lookup(full_idx)
                
        try:
            res_wav = audiodec.decoder.decode(zq)
            res_wav = res_wav.squeeze(1).transpose(1, 0).cpu().numpy()
        except:
            print('error in reconstruction')

    sf.write(
            './test/syn.wav',
            res_wav,
            sample_rate,
            "PCM_16",)

        
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        type=str,
                        default='config/hparams.yaml',
                        help='config path')
    
    parser.add_argument('--ar_ckpt',
                        type=str,
                        default='ckpt/basic/ar.pt',
                        help='AR ckpt path')

    parser.add_argument('--nar_ckpt',
                        type=str,
                        default='ckpt/basic/nar.pt',
                        help='NAR ckpt path')
    parser.add_argument('--prompt_wav',
                        type=str,
                        default='test/prompt_wavs/test_1.wav',
                        help='out_dir')
    parser.add_argument('--prompt_text',
                    type=str,
                    default='在夏日阴凉的树荫下，鸭妈妈孵着鸭宝宝。',
                    help='out_dir')
    parser.add_argument('--text',
                    type=str,
                    default='这一段戏同样也表达了亚瑟说的，我原本以为我的人生是一出悲剧，但其实它是一出戏剧。',
                    help='out_dir')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='cpu or cuda')

    args = parser.parse_args()
    hp = get_config_from_file(args.config).hparams

    do_tts(hp,args)