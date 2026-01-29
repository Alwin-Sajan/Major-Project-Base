import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from .config import CONVNEXT_MODEL_PATH



class ConvNeXtIncremental(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        in_features = backbone.classifier[2].in_features 
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        emb = self.proj(x)
        return F.normalize(emb, p=2, dim=1)

# ---- Embedding model ----
#===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CONVNEXT_MODEL_PATH, map_location=device)
prototypes = checkpoint["prototypes"].to(device)  
class_names = checkpoint["classes"]
num_classes = len(class_names)

embedding_net = ConvNeXtIncremental(
    embedding_dim=256,
    pretrained=False
)

embedding_net.load_state_dict(checkpoint["model_state"])
embedding_net.to("cuda").eval()

embedding_PRETRAINED_DEFAULT = ConvNeXtIncremental(embedding_dim=256,pretrained=True)
embedding_PRETRAINED_DEFAULT.to("cuda").eval()