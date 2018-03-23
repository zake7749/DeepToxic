from sklearn import metrics


def roc_auc_score(y, y_pred):
    return metrics.roc_auc_score(y, y_pred)


def log_loss(y, y_pred):
    total_loss = 0
    for j in range(6):
        loss = metrics.log_loss(y[:, j], y_pred[:, j])
        total_loss += loss
    total_loss /= 6.

    return total_loss