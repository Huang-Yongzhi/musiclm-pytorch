import torch
from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import FineTransformer, FineTransformerTrainer
from audiolm_pytorch import AudioLMSoundStream, AudioLM
import gc  # 导入垃圾回收模块
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import librosa
import numpy as np


nltk.download('punkt')

# 公共变量
checkpoint_path = 'hubert_base_ls960.pt'
kmeans_path = 'hubert_base_ls960_L9_km500.bin'

audio_output_dir = './downloaded_audios'
batch_size = 1
data_max_length = 320 * 32
num_train_steps = 1


def load_audio_file(file_path, sr=22050):
    # 加载音频文件
    # 'sr' 是采样率，22050 是常用值
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def audio_to_melspectrogram(audio, sr=22050, n_mels=128, hop_length=512):
    # 将音频转换为梅尔频谱图
    melspec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    return melspec


def process_audio_files(file_paths, sr=22050, n_mels=128, hop_length=512):
    processed_audios = []
    valid_indices = []  # 用于存储有效音频的索引

    for index, file_path in enumerate(file_paths):
        try:
            # 尝试加载和预处理音频文件
            audio = load_audio_file(file_path, sr=sr)
            melspec = audio_to_melspectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
            processed_audios.append(melspec)
            valid_indices.append(index)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return processed_audios, valid_indices



# 简单的文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 基本标记化
    tokens = word_tokenize(text)
    return tokens


# 函数：训练 FineTransformer
def train_fine_transformer(audio_data, combined_data):
    soundstream = AudioLMSoundStream()

    fine_transformer = FineTransformer(
                    num_coarse_quantizers=4, 
                    num_fine_quantizers=8, 
                    codebook_size=1024, 
                    dim=1024, 
                    depth=6, 
                    audio_text_condition=True # 需要输入文本
                    ).cuda()
    
    # 确保 Trainer 接收文本数据作为输入
    trainer = FineTransformerTrainer(
        transformer=fine_transformer,
        codec=soundstream, 
        audio_data=audio_data,
        text_data=combined_data, 
        folder=audio_output_dir, 
        batch_size=batch_size, 
        data_max_length=data_max_length, 
        num_train_steps=num_train_steps
        )    
    
    trainer.train()
    torch.save(fine_transformer.state_dict(), 'fine_transformer.pth')
    print("save fine_transformer.pth")    
    del fine_transformer, trainer, soundstream
    gc.collect()

# 加载 .csv 文件
csv_file = 'musiccaps-public.csv'
df = pd.read_csv(csv_file)

# 提取字段作为文本数据

captions = df['caption']
aspect_list = df['aspect_list']

# 应用预处理
preprocessed_captions = [preprocess_text(text) for text in captions]
preprocessed_aspects = [preprocess_text(text) for text in aspect_list]

#  并行结合
combined_data = list(zip(preprocessed_captions, preprocessed_aspects))



# 提取音频文件名
audio_filenames = df['ytid'].tolist()
# 构建音频文件的完整路径
audio_file_paths = [f'./downloaded_audios/{filename}.wav' for filename in audio_filenames]
# 预处理音频数据，并获取有效音频文件的索引
audio_data, valid_audio_indices = process_audio_files(audio_file_paths)

# 根据有效音频文件的索引来同步文本数据
synced_text_data = [combined_data[i] for i in valid_audio_indices]


# 确保长度相同, audio_data 和 combined_data 是一一对应的
assert len(audio_data) == len(synced_text_data)



train_fine_transformer(audio_data, synced_text_data)
