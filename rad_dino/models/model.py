import torch.nn as nn 

class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0] 
        # Equivalent: cls_token = self.backbone.get_intermediate_layers(x, n=1)[0][:, 0]  
        return self.head(cls_token)
    
if __name__ == "__main__":
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained('facebook/dinov2-base')
    # backbone = AutoModel.from_pretrained("microsoft/rad-dino")
    model = DinoClassifier(backbone, num_classes=10)
    for name, _ in model.named_parameters():
        print(f"Parameter name: {name}")