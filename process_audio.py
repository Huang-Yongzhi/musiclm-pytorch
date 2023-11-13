import librosa

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
    for file_path in file_paths:
        audio = load_audio_file(file_path, sr=sr)
        melspec = audio_to_melspectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        processed_audios.append(melspec)
    return processed_audios


