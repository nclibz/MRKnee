# %%
from typing import Union

from monai.networks.nets.efficientnet import EfficientNet, efficientnet_params


# %%
class EfficientNetBN(EfficientNet):
    def __init__(
        self,
        model_name: str,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        drop_rate: float = 0.2,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
    ) -> None:
        """
        Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
        model_name is mandatory argument as there is no EfficientNetBN itself,
        it needs the N in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model

        Args:
            model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
                norm is used.
            progress: whether to show download progress for pretrained weights download.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            norm: feature normalization type and arguments.
            adv_prop: whether to use weights trained with adversarial examples.
                This argument only works when `pretrained` is `True`.

        Examples::

            # for pretrained spatial 2D ImageNet
            >>> image_size = get_efficientnet_image_size("efficientnet-b0")
            >>> inputs = torch.rand(1, 3, image_size, image_size)
            >>> model = EfficientNetBN("efficientnet-b0", pretrained=True)
            >>> model.eval()
            >>> outputs = model(inputs)

            # create spatial 2D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=2)

            # create spatial 3D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

            # create EfficientNetB7 for spatial 2D
            >>> model = EfficientNetBN("efficientnet-b7", spatial_dims=2)

        """
        # block args
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params.keys():
            raise ValueError(
                "invalid model_name {} found, must be one of {} ".format(
                    model_name, ", ".join(efficientnet_params.keys())
                )
            )

        # get network parameters
        (
            weight_coeff,
            depth_coeff,
            image_size,
            dropout_rate,
            dropconnect_rate,
        ) = efficientnet_params[model_name]

        # override dropout rate
        dropout_rate = drop_rate

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
        )


# %%
