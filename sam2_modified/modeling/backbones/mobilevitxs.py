import torch
import torch.nn as nn
from sam2.modeling.backbones.custom_mobilevit import MobileViTModel
from transformers import MobileViTConfig

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MobileVitXS(nn.Module):
    def __init__(self, model_name: str = "apple/mobilevit-x-small", exportable: bool = False, pretrained: bool = False): ##out_chans: int = 256, img_size: int = 1024
        super().__init__()
        if pretrained:
            print(f"Loading MobileVitXS from checkpoint")
            self.model = MobileViTModel.from_pretrained(model_name, output_hidden_states=True, exportable=exportable, ignore_mismatched_sizes=True)
        else:
            print(f"Initializing MobileVitXS from scratch")
            config = MobileViTConfig.from_pretrained(model_name)
            config.output_hidden_states = True
            self.model = MobileViTModel(config=config, exportable=exportable)

        self.channel_list = [80, 64, 48]
        
        # self.out_chans = out_chans
        # self.img_size = img_size

        # Project MobileViT's embedding dim (e.g., 384) to match out_chans like ImageEncoderViT
        # in_features = 384
        # self.neck = nn.Sequential(
        #     nn.Conv2d(in_features, out_chans, kernel_size=1, bias=False),
        #     LayerNorm2d(out_chans),  # LayerNorm2d-like behavior
        #     nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        #     LayerNorm2d(out_chans),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, C, H, W)

        Returns:
            Tensor: shape (B, out_chans, H_feat, W_feat)
        """
                
        outputs = self.model(pixel_values=x)
        hidden_state = outputs.last_hidden_state  # shape: (B, C, H, W)
        
        all_hidden_states = outputs.hidden_states[1:]  # shape: (B, C, H, W)
                
        return all_hidden_states
        
        # feat = self.neck(hidden_state)
        
        # return feat

    def get_num_layers(self) -> int:
        return len(self.channel_list)

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("conv_stem") != -1:
            return 0
        elif layer_name.find("layer") != -1:
            return int(layer_name.split("layer")[1].split(".")[1]) + 1
        else:
            return num_layers + 1