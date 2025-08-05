import os
import torch
import matplotlib.pyplot as plt
from lxt.efficient import monkey_patch

def visualize_lrp_maps(
    model: torch.nn.Module,
    images: torch.Tensor,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    image_ids: list[str],
    output_dir: str,
    multi_view: bool = False,
):
    """
    Compute and save LRP (Gradient * Input) relevance overlays.

    Args:
        model:            Your DinoClassifier (in eval mode).
        images:           [B,C,H,W] or [B,4,C,H,W] pre-normalized tensors.
        image_mean:       Mean of the image processor.
        image_std:        Standard deviation of the image processor.
        image_ids:        List of B identifiers for filenames.
        output_dir:       Directory to save lrp_*.png.
        multi_view:       Whether `images` has 4-view layout.
    """
    device = next(model.parameters()).device
    model.eval()
    # Monkey-patch the model to inject LRP hooks
    monkey_patch(model, verbose=True)

    # forward + backward
    imgs = images.to(device).requires_grad_()  # keep grads on input
    logits, _ = model(imgs)
    # build one-hot for top-1 of each sample
    top1 = logits.argmax(dim=-1)
    onehot = torch.zeros_like(logits).to(device)
    for i, t in enumerate(top1):
        onehot[i, t] = 1.0
    logits.backward(gradient=onehot)

    grads = imgs.grad  # same shape as imgs

    # compute per-pixel relevance: sum over channels
    if multi_view:
        # imgs: [B,4,C,H,W] → rel: [B,4,H,W]
        rel = (imgs * grads).sum(dim=2)
        norm = rel.abs().amax(dim=(2,3), keepdim=True).clamp(min=1e-8)
        heatmaps = (rel / norm).detach().cpu().numpy()
        denorm = (imgs * image_std + image_mean).detach().cpu().numpy()
    else:
        # imgs: [B,C,H,W] → rel: [B,H,W]
        rel = (imgs * grads).sum(dim=1)
        norm = rel.abs().amax(dim=(1,2), keepdim=True).clamp(min=1e-8)
        heatmaps = (rel / norm).detach().cpu().numpy()
        denorm = (imgs * image_std + image_mean).detach().cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    # overlay & save
    B = heatmaps.shape[0]
    for i in range(B):
        if multi_view:
            view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
            V = heatmaps.shape[1]
            assert V == len(view_names), f"NotEqualError: V: {V}, expected: {len(view_names)}"
            for v in range(V):
                img = denorm[i,v].transpose(1,2,0)
                hm  = heatmaps[i, v]
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(img)
                ax.imshow(hm, cmap='seismic', alpha=0.5, vmin=-1, vmax=1)
                ax.axis('off')
                fig.savefig(os.path.join(output_dir, f"lrp_{image_ids[i]}_{view_names[v]}.png"),
                            bbox_inches='tight', pad_inches=0)
                plt.close(fig)
        else:
            img = denorm[i].transpose(1,2,0)
            hm  = heatmaps[i]
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(img)
            ax.imshow(hm, cmap='seismic', alpha=0.5, vmin=-1, vmax=1)
            ax.axis('off')
            fig.savefig(os.path.join(output_dir, f"lrp_{image_ids[i]}.png"),
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)
