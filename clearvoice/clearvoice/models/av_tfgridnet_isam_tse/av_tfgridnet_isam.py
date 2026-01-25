import os
import sys
import torch
import torch.nn as nn

# Add the training model path to sys.path
train_model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'train', 'target_speaker_extraction')
sys.path.insert(0, train_model_path)

from models.av_tfgridnetV3_isam.av_tfgridnetv3_isam import av_TFGridNetV3_isam
from models.visual_frontend.resnet18 import Visual_encoder

class AV_TFGridNet_ISAM_TSE_16K(nn.Module):
    """
    Audio-Visual TFGridNet with ISAM (Inter-Speaker Attention Module) for target speaker extraction at 16 kHz.

    This model processes videos of 2 speakers simultaneously and separates their voices.
    It uses visual cues (lip movements) to guide the audio separation.

    Args:
        args: Argument parser object containing model configuration settings.
    """

    def __init__(self, args):
        super(AV_TFGridNet_ISAM_TSE_16K, self).__init__()
        self.args = args

        # Visual encoder for extracting lip features
        self.ref_encoder = Visual_encoder(args)

        # Audio separation network with visual guidance
        self.sep_network = av_TFGridNetV3_isam(args)

    def forward(self, mixture, visual):
        """
        Args:
            mixture: [B, T], audio mixture with B batch size and T samples
            visual: [B, speaker_no, frames, 112, 112], visual features for each speaker
        Returns:
            est_source: [B, speaker_no, T], separated source for each speaker
        """
        # print('actual AV_TFGridNet_ISAM_TSE_16K forward')
        # print('mixture:', mixture.shape)
        # print('visual:', visual.shape)
        # Process visual input through encoder
        # visual shape: [B, speaker_no, frames, 112, 112]
        visual = visual.to(self.args.device)
        batch_size = visual.size(0)
        speaker_no = visual.size(1)

        # Reshape to process all speakers together
        visual = visual.view(batch_size * speaker_no, visual.size(2), 112, 112)
        ref = self.ref_encoder(visual)  # [B*speaker_no, emb_size, frames]

        # Separation network
        est_source = self.sep_network(mixture, ref) # [B, speaker_no, frames]

        return est_source
