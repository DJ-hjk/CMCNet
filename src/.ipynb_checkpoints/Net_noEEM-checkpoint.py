import torch
import torch.nn as nn

from src.cnn_change import CNN
from src.newfinalfusion_change import fusion
from src.Doconv import DOConv2d
from src.SwinUMamba import SwinUMamba
from src.decoder_mamba import UNetResDecoder
import torch.nn.functional as F
class Net_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = CNN()
        self.encoder2 = SwinUMamba()
        self.fusion1 = fusion(64,64)
        self.fusion2 = fusion(128,128)
        self.fusion3 = fusion(256,256)
        self.fusion4 = fusion(512,512)
        self.decoder_mamba = UNetResDecoder()

    def forward(self, x):
        x1, x2, x3, x4, out1 = self.encoder1(x)
        y1, y2, y3, y4, out2 = self.encoder2(x)
        
        f1 = self.fusion1(x2, y2)
        f2 = self.fusion2(x3, y3)
        f3 = self.fusion3(x4, y4)
        out = self.fusion4(out1,out2)
        mask_mamba = self.decoder_mamba(f1,f2,f3,out)

        return mask_mamba

