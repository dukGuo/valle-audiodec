# VALL-E 

Inference code for [Wenetspeech4TTS/Audiodec-Valle-Wenetspeech4TTS](https://huggingface.co/Wenetspeech4TTS/Audiodec-Valle-Wenetspeech4TTS)

## Installation

  ``` bash
  git clone 
  cd valle-audiodec
  pip install -r requirements.txt
  ```

## Download pre-train model
### AudioDec
We use [AudioDec](https://github.com/facebookresearch/AudioDec/) as our speech tokenizer to further improve audio quality.

Please download the whole [exp](https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip) folder, unzip and put it in the `AudioDec/exp` directory.

```bash
cd valle-audiodec
wget https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip
unzip exp.zip
mv exp AudioDec/exp
```

### VALL-E
  Checkpiont available on [Wenetspeech4TTS/Audiodec-Valle-Wenetspeech4TTS](https://huggingface.co/Wenetspeech4TTS/Audiodec-Valle-Wenetspeech4TTS)

- VALL-E *Basic* :VALL-E trained with the WenetSpeech4TTS Basic subset
- VALL-E *Standard*: VALL-E *Basic* fine-tuning with the WenetSpeech4TTS Standard subset
- VALL-E *Premium*: VALL-E *Standard* fine-tuning with the WenetSpeech4TTS Premium subset
## Speech Sample

https://wenetspeech4tts.github.io/wenetspeech4tts

https://rxy-j.github.io/HPMD-TTS

## Inference

``` bash
  cd valle-audiodec
  python infer_tts.py \ 
    --config config/hparams.yaml \
    --ar_ckpt ckpt/basic/ar.pt \
    --nar_ckpt ckpt/basic/nar.pt \
    --prompt_wav test/prompt_wavs/test_1.wav \
    --prompt_text 在夏日阴凉的树荫下，鸭妈妈孵着鸭宝宝。 \
    --text 负责指挥的将军在一旁交代着注意事项，每个人在上面最多只能待九十秒。
```

> To improve audio quality and ensure consistent volume levels across different inputs, it is advisable to normalize the loudness of the prompt waveform before conducting inference. This preprocessing step helps achieve uniformity in the audio input, which can lead to more reliable inference outcomes.
> ```
> sox $in_wave -r $sample_rate -b 16 --norm=-6 $out_wave
> ```

## References
This repository is developed based on the following repositories.

- [facebookresearch/AudioDec](https://github.com/facebookresearch/AudioDec)
- [lifeiteng/vall-e](https://github.com/lifeiteng/vall-e)
- [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
