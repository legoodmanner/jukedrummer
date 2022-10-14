from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from .LanguageModel import PositionalEncoding

class ConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        modelseq = [
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, stride=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=8),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, stride=8),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=16, stride=1),
            nn.ReLU(inplace=True),

        ]
        self.model = nn.Sequential(*modelseq)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        if x.size(2) == 1:
            x = x.permute(0,2,1)
        r = self.model(x)
        r = torch.mean(r, dim=-1)
        return r

class TransformerDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(2048, 512)
        self.pos_enc = PositionalEncoding(512, max_len=10000)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(512,8),
            num_layers=4,
        )
        self.output_map = nn.Linear(512, 1)
        self.feature_map = nn.Linear(4800, 512)

    def forward(self, tgz, ot):
        """
        Args:
            tgz: shape=(bs, 8192)
            ot: shape=(bs, 8192, 4800), pretrained from jukebox
        """
        tg = self.embed(tgz)
        tg = self.pos_enc(tg)
        tg = self.model(tg, self.feature_map(ot))
        return self.output_map(x)

if __name__ == '__main__':
    import soundfile as sf
    model = ConvDiscriminator()
    wav, sr = sf.read('/home/lego/NAS189/home/codify/data/drums/audio_24s/others/Zeno - Signs_5.wav')
    print(wav.shape)
    x = model(torch.FloatTensor(wav.mean(-1)[np.newaxis,:]))
    print(x)