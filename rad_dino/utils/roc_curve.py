import numpy as np 
from sklearn.metrics import roc_curve, auc

def compute_stats(stats, ci=95):
    mean = np.mean(stats)
    lower_bound = np.percentile(stats, (100 - ci) / 2)
    upper_bound = np.percentile(stats, 100 - (100 - ci) / 2)
    std_dev = np.std(stats)
    return mean, (lower_bound, upper_bound), std_dev

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, ci=95, seed=42):
    """Computes bootstrap confidence intervals for a given metric function."""
    np.random.seed(seed)  # Set the seed for reproducibility
    stats = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)  # Resampling with replacement
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        stats.append(metric_func(y_true_resampled, y_pred_resampled))

    return compute_stats(stats, ci)

def auc_bootstrapping(y_true, y_score, bootstrapping=1000, drop_intermediate=False):
    tprs, aucs, thrs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    np.random.seed(0)
    rand_idxs = np.random.randint(0, len(y_true), size=(bootstrapping, len(y_true))) # Note: with replacement 
    for rand_idx in rand_idxs:
        y_true_set = y_true[rand_idx]
        y_score_set = y_score[rand_idx]
        fpr, tpr, thresholds = roc_curve(y_true_set, y_score_set, drop_intermediate=drop_intermediate)
        tpr_interp = np.interp(mean_fpr, fpr, tpr) # must be interpolated to gain constant/equal fpr positions
        tprs.append(tpr_interp) 
        aucs.append(auc(fpr, tpr))
        optimal_idx = np.argmax(tpr - fpr)
        thrs.append(thresholds[optimal_idx])
    return tprs, aucs, thrs, mean_fpr

def cm2acc(cm):
    # [[TN, FP], [FN, TP]] 
    tn, fp, fn, tp = cm.ravel()
    return (tn+tp)/(tn+tp+fn+fp)

def safe_div(x,y):
    if y == 0:
        return float('nan') 
    return x / y

def cm2x(cm):
    tn, fp, fn, tp = cm.ravel()
    pp = tp + fp  # predicted positive 
    pn = fn + tn  # predicted negative
    p = tp + fn   # actual positive
    n = fp + tn   # actual negative  

    ppv = safe_div(tp,pp)  # positive predictive value 
    npv = safe_div(tn,pn)  # negative predictive value 
    tpr = safe_div(tp,p)   # true positive rate (sensitivity, recall)
    tnr = safe_div(tn,n)   # true negative rate (specificity)
    # Note: other values are 1-x eg. fdr=1-ppv, for=1-npv, ....
    return ppv, npv, tpr, tnr