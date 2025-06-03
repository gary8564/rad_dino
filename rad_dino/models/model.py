import torch
import torch.nn as nn 

class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        embed_dim = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        outputs = self.backbone(
            x,
            output_attentions=True,
            return_dict=True,
        )
        cls_token = outputs.last_hidden_state[:, 0] 
        # Equivalent: cls_token = self.backbone.get_intermediate_layers(x, n=1)[0][:, 0]  
        logits = self.head(cls_token)   
        # Also return all attention maps (as a tuple of length num_layers)
        # Each attentions[i] has shape (B, num_heads, seq_len, seq_len).
        attentions = outputs.attentions 
        # ONNX doesn't like Python lists, so we'll stack them into one large tensor:
        # stacked_attns.shape == (num_layers, B, num_heads, seq_len, seq_len)
        stacked_attns = torch.stack(attentions, dim=0)
        return logits, stacked_attns
    
if __name__ == "__main__":
    from transformers import AutoModel
    
    def unfreeze_layers(model, num_unfreeze_layers):
        num_total_layers = model.backbone.config.num_hidden_layers
        assert num_total_layers == 12, "Number of total layers is not 12" # 12 layers total for ViT-B-14
        assert num_unfreeze_layers <= num_total_layers, "Number of unfreeze layers cannot be greater than the total number of layers"
        # First freeze all backbone parameters
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        # Then unfreeze the specified layers
        for i in range(num_total_layers - 1, num_total_layers - num_unfreeze_layers - 1, -1):
            for name, param in model.backbone.named_parameters():
                if f"layer.{i}" in name:
                    param.requires_grad = True
                    
    backbone = AutoModel.from_pretrained('facebook/dinov2-base')
    # backbone = AutoModel.from_pretrained("microsoft/rad-dino")
    model = DinoClassifier(backbone, num_classes=10)
    # Get the number of hidden layers (transformer blocks) for ViT-B-14
    num_layers = model.backbone.config.num_hidden_layers
    print(f"Number of transformer blocks: {num_layers}")
    unfreeze_layers(model, 2)
    for name, param in model.named_parameters():
        if 'backbone' in name:
            if param.requires_grad:
                print(f"Parameter name: {name}")
        else:
            param.requires_grad = True
            print(f"Parameter name: {name}")
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")