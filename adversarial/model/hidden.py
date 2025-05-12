import torch
import torch.nn as nn
import random
import numpy as np

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser

from model.encoder import Encoder
from model.decoder import Decoder
from model.resnet18 import ResNet

import sys

sys.path.append("..")

from noise_layers.identity import Identity
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.brightness import Brightness
from noise_layers.gaussian_blur import GaussianBlur


###############################################################################
# 1) Placeholder base class for JpegMask (replace with your real JpegBasic if you have it)
###############################################################################
class JpegBasic(nn.Module):
    """
    Placeholder base class for JPEG transformations.
    You should replace 'yuv_dct' and 'idct_rgb' with your real code that:
      - converts from [-1,1] to [0,255],
      - does RGB->YUV,
      - blockwise DCT,
      - IDCT,
      - YUV->RGB,
      - [0,255]->[-1,1], etc.
    """

    def __init__(self):
        super(JpegBasic, self).__init__()

    def yuv_dct(self, image, subsample=0):
        # TODO: Replace with your actual logic
        return image, 0, 0

    def idct_rgb(self, dct_image, pad_width, pad_height):
        # TODO: Replace with your actual logic
        return dct_image


###############################################################################
# 2) JpegMask: Masks out higher-frequency DCT components
###############################################################################
class JpegMask(JpegBasic):
    def __init__(self, Q=50, subsample=0):
        super(JpegMask, self).__init__()
        self.Q = Q
        # Example usage of Q for some scaling
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q
        self.subsample = subsample

    def round_mask(self, x):
        """
        Zero out some higher-frequency coefficients in the DCT domain.
        x shape: (N, C, H, W) after blockwise DCT (H,W multiple of 8).
        """
        mask = torch.zeros(1, 3, 8, 8, device=x.device)
        # Example: keep a 5x5 block of low frequencies for the 1st channel,
        # and 3x3 for the 2nd and 3rd channels
        mask[:, 0:1, :5, :5] = 1
        mask[:, 1:3, :3, :3] = 1

        # repeat mask across the entire (H//8, W//8) blocks
        mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
        return x * mask

    def forward(self, image_and_cover):
        """
        Expects (image, cover_image) but we only use 'image' here.
        """
        image, cover_image = image_and_cover
        # 1) Convert to YUV+DCT domain
        image_dct, pad_w, pad_h = self.yuv_dct(image, self.subsample)
        # 2) Apply the round mask
        masked_dct = self.round_mask(image_dct)
        # 3) IDCT -> clamp
        noised = self.idct_rgb(masked_dct, pad_w, pad_h).clamp(-1, 1)
        return noised


###############################################################################
# 3) MBRS: Randomly picks from [JpegMask(50), DiffJPEG(50), Identity()]
###############################################################################
class MBRS(nn.Module):
    """
    On each call, MBRS picks a random sub-noise:
       0 -> JpegMask(50)
       1 -> DiffJPEG(50)
       2 -> Identity()

    Supports a per-image noise option.
    """

    def __init__(self, device):
        super(MBRS, self).__init__()
        self.jpeg_mask = JpegMask(Q=50)
        self.diff_jpeg = DiffJPEG(quality=50, device=device)
        self.identity = Identity()

    def forward(self, image, cover_image=None, per_image_noise=False):
        if per_image_noise:
            # Apply noise separately for each image in the batch.
            noised_images = []
            for i in range(image.size(0)):
                choice = torch.randint(0, 3, (1,)).item()
                single_image = image[i:i + 1]
                # Use corresponding cover if provided; otherwise, fallback.
                if cover_image is not None:
                    single_cover = cover_image[i:i + 1]
                else:
                    single_cover = single_image

                if choice == 0:
                    noised_images.append(self.jpeg_mask((single_image, single_cover)))
                elif choice == 1:
                    noised_images.append(self.diff_jpeg(single_image))
                else:
                    noised_images.append(self.identity(single_image))
            return torch.cat(noised_images, dim=0)
        else:
            # Apply one noise to the entire batch.
            choice = torch.randint(0, 3, (1,)).item()
            if choice == 0:
                if cover_image is None:
                    cover_image = image
                return self.jpeg_mask((image, cover_image))
            elif choice == 1:
                return self.diff_jpeg(image)
            else:
                return self.identity(image)


###############################################################################
# 4) The Hidden class with MBRS integrated (plus your original noise)
###############################################################################
class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device,
                 noiser: Noiser, model_type: str):
        """
        :param configuration: Configuration for the net (image size, channels, etc.)
        :param device: torch.device (CPU or GPU)
        :param noiser: a Noiser object (used for fallback if epoch < 50)
        :param model_type: 'cnn' or 'resnet' for the decoder
        """
        super(Hidden, self).__init__()

        # -----------------------------
        # Encoder + Decoder
        # -----------------------------
        self.encoder = Encoder(configuration).to(device)
        self.noiser_train = noiser  # used if needed when epoch < 50
        self.noiser_test = Identity()  # default for validation

        if model_type == 'cnn':
            self.decoder = Decoder(configuration).to(device)
        elif model_type == 'resnet':
            self.decoder = ResNet(configuration).to(device)

        # -----------------------------
        # Discriminator
        # -----------------------------
        self.discriminator = Discriminator(configuration).to(device)

        # -----------------------------
        # Optimizers
        # -----------------------------
        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        # -----------------------------
        # Optional VGG Loss
        # -----------------------------
        self.vgg_loss = VGGLoss(3, 1, False).to(device) if configuration.use_vgg else None

        # Store config and device
        self.config = configuration
        self.device = device

        # Basic losses
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Discriminator labels
        self.cover_label = 1
        self.encoded_label = 0

        # Optional TB logger
        self.tb_logger = None

        # -----------------------------
        # Original noise layers (0..4) for "hidden" approach
        # -----------------------------
        self.noise_list = [0, 1, 2, 3, 4]
        self.noises = {
            0: Identity(),
            1: DiffJPEG(50, device),  # randomize 'quality' below
            2: Gaussian(0.0),  # randomize 'noise_var'
            3: GaussianBlur(std=1.0),  # randomize 'std'
            4: Brightness(1.0)  # randomize 'factor'
        }

        # -----------------------------
        # MBRS noise (random among JpegMask, DiffJPEG, Identity)
        # -----------------------------
        self.mbrs = MBRS(device)

    def _randomize_noise_params(self, noise_idx):
        """
        Randomize parameters for the selected noise layer (only needed for some).
        Called once per batch if using the "hidden" approach with epoch >= 50.
        """
        if noise_idx == 1:
            # DiffJPEG
            self.noises[1].quality = random.randint(50, 99)
        elif noise_idx == 2:
            # Gaussian
            self.noises[2].noise_var = random.uniform(0, 0.1)
        elif noise_idx == 3:
            # GaussianBlur
            self.noises[3].std = random.uniform(0, 1.0)
        elif noise_idx == 4:
            # Brightness
            self.noises[4].factor = random.uniform(1.0, 3.0)
        # if 0 -> Identity, no params needed

    def train_on_batch(self, batch: list, epoch, mbrs=False, per_image_noise=False):
        """
        Trains the network on a single batch: (images, messages).

        - If mbrs=True, apply MBRS noise (one random choice among JpegMask, DiffJPEG, Identity)
          once for the entire batch, or per image if per_image_noise=True.

        - If mbrs=False:
            * If epoch < 50 -> fallback to self.noiser_train for the entire batch.
            * If epoch >= 50:
                + If per_image_noise=True -> pick a random noise for each sample in the batch.
                + Else (the default) -> pick one random noise for the entire batch.
        """
        images, messages = batch
        batch_size = images.shape[0]

        # Switch to train mode
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            # ---------- 1) Train Discriminator ------------
            self.optimizer_discrim.zero_grad()

            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            # Discriminator on real images
            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()

            # Encode the images with the hidden message
            encoded_images = self.encoder(images, messages)

            # ------------------------------------------------
            # Pick the noise approach
            # ------------------------------------------------
            if mbrs:
                if epoch < 50:
                    noised_images = self.noiser_train(encoded_images)
                else:
                    # Use MBRS noise with an option for per-image noise.
                    noised_images = self.mbrs(encoded_images, cover_image=images, per_image_noise=per_image_noise)
            else:
                # The "hidden" approach
                if epoch < 50:
                    noised_images = self.noiser_train(encoded_images)
                else:
                    if per_image_noise:
                        noised_batch = []
                        for i in range(batch_size):
                            single_encoded = encoded_images[i].unsqueeze(0)  # (1, C, H, W)
                            choice = random.choice(self.noise_list)  # e.g. [0..4]
                            self._randomize_noise_params(choice)  # randomize params
                            noised_sample = self.noises[choice](single_encoded)
                            noised_batch.append(noised_sample)
                        noised_images = torch.cat(noised_batch, dim=0)  # (batch_size, C, H, W)
                    else:
                        choice = random.choice(self.noise_list)
                        self._randomize_noise_params(choice)
                        noised_images = self.noises[choice](encoded_images)
            # ------------------------------------------------

            # Discriminator on encoded images
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())
            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # ---------- 2) Train Encoder + Decoder ---------
            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            # Generator adversarial loss
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            # Encoder loss (MSE or VGG)
            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # Decoder loss (MSE on messages)
            decoded_messages = self.decoder(noised_images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)

            # Total generator loss
            g_loss = (self.config.adversarial_loss * g_loss_adv +
                      self.config.encoder_loss * g_loss_enc +
                      self.config.decoder_loss * g_loss_dec)
            g_loss.backward()
            self.optimizer_enc.step()
            self.optimizer_dec.step()

        # Bitwise accuracy (assuming messages are in {0,1})
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.mean(np.abs(decoded_rounded - messages.detach().cpu().numpy()))
        bitwise_acc = 1.0 - bitwise_avg_err

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-acc    ': bitwise_acc,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, decoded_messages)

    def validate_on_batch(self, batch: list, test_noiser='Identity'):
        """
        Validation on a single batch: (images, messages).
        By default uses Identity for noise, but you can pass e.g. 'DiffJPEG' if you want.
        """
        images, messages = batch
        batch_size = images.shape[0]

        # Switch to Identity if requested
        if test_noiser == 'Identity':
            self.noiser_test = Identity()
        # (You could add logic to pick other test noises if needed.)

        # Eval mode
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        # Optional TB logging
        if self.tb_logger is not None:
            encoder_final = self.encoder._modules['final_layer']
            decoder_final = self.decoder._modules['linear']
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            # Discriminator on cover
            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            # Encode
            encoded_images = self.encoder(images, messages)

            # Noise (Identity by default)
            noised_images = self.noiser_test(encoded_images)

            # Decode
            decoded_messages = self.decoder(noised_images)

            # Discriminator on encoded images
            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            # Generator adversarial loss
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            # Encoder loss (MSE or VGG)
            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # Decoder loss (MSE on messages)
            g_loss_dec = self.mse_loss(decoded_messages, messages)

            # Total loss
            g_loss = (self.config.adversarial_loss * g_loss_adv +
                      self.config.encoder_loss * g_loss_enc +
                      self.config.decoder_loss * g_loss_dec)

        # Bitwise accuracy
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.mean(np.abs(decoded_rounded - messages.detach().cpu().numpy()))
        bitwise_acc = 1.0 - bitwise_avg_err

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-acc    ': bitwise_acc,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}\n{}'.format(str(self.encoder), str(self.decoder), str(self.discriminator))
