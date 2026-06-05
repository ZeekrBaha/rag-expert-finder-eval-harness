def confusion_matrix(y_true: list, y_pred: list) -> dict:
    tp = fn = tn = fp = 0
    for t, p in zip(y_true, y_pred):
        t, p = bool(t), bool(p)
        if t and p:
            tp += 1
        elif t and not p:
            fn += 1
        elif not t and p:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def precision_recall(y_true: list, y_pred: list) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two raters over binary labels."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected agreement by chance
    pe = 0.0
    for label in set(a) | set(b):
        pa = sum(1 for x in a if x == label) / n
        pb = sum(1 for y in b if y == label) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
