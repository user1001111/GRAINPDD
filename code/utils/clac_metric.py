import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def sk_metric(y_label, y_pred):
    # AUPR AUC
    return [round(average_precision_score(y_label, y_pred),4), 
            round(roc_auc_score(y_label, y_pred),4), 3, 4, 5, 6, 7]
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def get_metrics(real_score, predict_score):
    # ===== Compute AUPR and AUC with sklearn, matching the training log =====
    y_true = np.array(real_score).flatten()
    y_pred = np.array(predict_score).flatten()
    aupr = average_precision_score(y_true, y_pred)   # Same computation as in testModel.
    auc  = roc_auc_score(y_true, y_pred)             # Same computation as in testModel.

    # ===== Preserve the original threshold search for F1, accuracy, recall, specificity, and precision =====
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    # The manual AUC calculation below was replaced by sklearn and is no longer used.
    # auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])  # Unused.

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    # The manual AUPR calculation is also no longer used.
    # aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])  # Unused.

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    # Preserve the original return order: [AUPR, AUC, F1, Accuracy, Recall, Specificity, Precision].
    return [aupr, auc, f1_score[0,0], accuracy[0,0], recall[0,0], specificity[0,0], precision[0,0]]


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)
