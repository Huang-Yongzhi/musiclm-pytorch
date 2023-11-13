import torch
from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import FineTransformer, FineTransformerTrainer
from audiolm_pytorch import AudioLMSoundStream, AudioLM
import gc  # 导入垃圾回收模块

# 公共变量
checkpoint_path = 'hubert_base_ls960.pt'
kmeans_path = 'hubert_base_ls960_L9_km500.bin'

audio_output_dir = './downloaded_audios'
batch_size = 1
data_max_length = 320 * 32
num_train_steps = 1

# 函数：训练 CoarseTransformer
def train_coarse_transformer():  
    wav2vec = HubertWithKmeans(checkpoint_path=checkpoint_path, kmeans_path=kmeans_path)   # 每个函数中重新创建 wav2vec，后面会删掉
    soundstream = AudioLMSoundStream()

    coarse_transformer = CoarseTransformer(num_semantic_tokens=wav2vec.codebook_size, codebook_size=1024, num_coarse_quantizers=4, dim=1024, depth=6, audio_text_condition=True).cuda()
    trainer = CoarseTransformerTrainer(transformer=coarse_transformer, codec=soundstream, wav2vec=wav2vec, audio_conditioner=quantizer, folder=audio_output_dir, batch_size=batch_size, data_max_length=data_max_length, num_train_steps=num_train_steps)
    trainer.train()
    torch.save(coarse_transformer.state_dict(), 'coarse_transformer.pth')
    del coarse_transformer, trainer, wav2vec, soundstream
    gc.collect()

train_coarse_transformer()