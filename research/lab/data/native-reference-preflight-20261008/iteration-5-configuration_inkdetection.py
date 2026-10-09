"""Configuration class for the Vesuvius ink-detection model.

See modeling_inkdetection.py for the model itself. Designed to be
loaded with `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
"""
from transformers import PretrainedConfig


class InkDetectionConfig(PretrainedConfig):
    """Configuration for an InkDetectionModel.

    Architecture: ResNet3D-50 backbone (Hara et al., 2018) initialised
    from Kinetics-700; per-stage max-pool over the z (depth) axis to
    collapse 5-D features to 4-D; small 2-D U-Net decoder; 1x1 conv
    output head producing one sigmoid logit channel.
    """
    model_type = "inkdetection_resnet3d"

    def __init__(
        self,
        in_channels: int = 1,
        input_depth: int = 62,
        input_size: int = 256,
        backbone_depth: int = 50,
        backbone_channels=(256, 512, 1024, 2048),
        num_classes: int = 1,
        decoder_upscale: int = 1,
        # Optional metadata (informational only)
        train_segment: str = "l_2",
        train_inklabels: str = "l_2_inklabels.png",
        train_steps: int = 12396,
        train_tiles: int = 0,
        train_loss_final: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.input_depth = input_depth
        self.input_size = input_size
        self.backbone_depth = backbone_depth
        self.backbone_channels = list(backbone_channels)
        self.num_classes = num_classes
        self.decoder_upscale = decoder_upscale
        self.train_segment = train_segment
        self.train_inklabels = train_inklabels
        self.train_steps = train_steps
        self.train_tiles = train_tiles
        self.train_loss_final = train_loss_final
