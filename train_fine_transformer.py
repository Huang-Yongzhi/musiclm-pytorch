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


# 函数：训练 FineTransformer
def train_fine_transformer():
    soundstream = AudioLMSoundStream()

    fine_transformer = FineTransformer(num_coarse_quantizers=4, num_fine_quantizers=8, codebook_size=1024, dim=1024, depth=6, audio_text_condition=True).cuda()
    trainer = FineTransformerTrainer(transformer=fine_transformer, codec=soundstream, folder=audio_output_dir, batch_size=batch_size, data_max_length=data_max_length, num_train_steps=num_train_steps)
    trainer.train()
    torch.save(fine_transformer.state_dict(), 'fine_transformer.pth')
    print("save fine_transformer.pth")    
    del fine_transformer, trainer, soundstream
    gc.collect()

train_fine_transformer()