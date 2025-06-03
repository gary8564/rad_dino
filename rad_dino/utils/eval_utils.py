from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score

def get_eval_metrics(task: str, num_classes: int, device: str):
    """Create appropriate metrics based on task type."""
    metrics = {}
    
    if task == "multiclass":
        metrics.update({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "top5_acc": Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
            "auroc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            "ap": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro"),
            "f1_score": F1Score(task="multiclass", num_classes=num_classes)
        })
    elif task == "multilabel":
        metrics.update({
            "acc": Accuracy(task="multilabel", num_labels=num_classes),
            "auroc": AUROC(task="multilabel", num_labels=num_classes, average="macro"),
            "ap": AveragePrecision(task="multilabel", num_labels=num_classes, average="macro"),
            "f1_score": F1Score(task="multilabel", num_labels=num_classes)
        })
    elif task == "binary":
        metrics.update({
            "acc": Accuracy(task="binary"),
            "top5_acc": Accuracy(task="binary", top_k=5),
            "auroc": AUROC(task="binary"),
            "ap": AveragePrecision(task="binary"),
            "f1_score": F1Score(task="binary")
        })
    
    return {k: v.to(device) for k, v in metrics.items() if v is not None}
