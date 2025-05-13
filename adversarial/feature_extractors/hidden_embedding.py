from model.hidden import (Hidden)
from options import *
import torch
from noise_layers.noiser import Noiser
import adversarial.utils_hidden as utils
from .base_encoder import BaseEncoder


class HiddenEmbedding(BaseEncoder):
    def __init__(self, message):
        super(HiddenEmbedding, self).__init__()
        hidden_config = HiDDenConfiguration(H=128, W=128,
                                            message_length=message,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3,
                                            enable_fp16=False
                                            )
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        noiser = Noiser([], device)
        self.model = Hidden(hidden_config, device, noiser, 'cnn')
        cp_file = f'/scratch/qilong3/transferattack/models/hidden/cnn30db/model_2.pth'  # always use model #1. on Madison it is the given model.

        vic_checkpoint = torch.load(cp_file, map_location='cpu')
        utils.model_from_checkpoint(self.model, vic_checkpoint)
        self.model = self.model.encoder.conv_layers
        # turn off gradients
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = 2.0 * images - 1.0
        return self.model(images)
