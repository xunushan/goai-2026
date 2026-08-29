import torch
import einops
import torch.nn as nn
from torchvision import transforms

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


def _load_dino_backbone(name):
    """加载 DINOv2 骨干（架构必须与训练一致：facebookresearch/dinov2@b48308a）。

    服务部署时权重由 ckpt 的 model_state_dict 灌回，因此优先用 pretrained=False
    纯架构加载，避免从 dl.fbaipublicfiles.com 下载 ~50MB 预训练权重（部分网络不可达）。
    仅当传入参数显式要求预训练权重（本服务不需要）时保留默认行为。
    """
    return torch.hub.load(
        "facebookresearch/dinov2:b48308a", name, pretrained=False
    )


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, output_dim=None, postprocess=None, n_patches=256):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        self.base_model = _load_dino_backbone(name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.output_dim = self.emb_dim # for compatibility
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

        self.postprocess = postprocess
        if postprocess is not None:
            if postprocess == 'avg_pool':
                self.latent_ndim = 1

        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalization = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def forward(self, x):
        # Accept arbitrary number of leading dimensions before (C, H, W)
        # and preserve them on return.
        # Example: input shape (...prefix, C, H, W)
        assert x.max() <= 1.0 and x.min() >= 0, "expect 0..1 range"
        x = self.normalization(x)

        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]

        # Collapse all leading dims into a single batch dimension for the base model
        prod_prefix = 1
        for d in prefix_shape:
            prod_prefix *= d
        x = x.reshape(prod_prefix, c, h, w)

        emb = self.base_model.forward_features(x)[self.feature_key]
        emb = emb.reshape(*prefix_shape, *emb.shape[1:])

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=-2)  # (...prefix, E)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape))

        return emb
