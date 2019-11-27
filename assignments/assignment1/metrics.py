import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp_fp_sum = sum(prediction == ground_truth)
    tp_sum = sum(prediction * ground_truth)
    true_sum = sum(ground_truth)
    pred_sum = sum(prediction)
    precision = tp_sum / pred_sum
    recall = tp_sum / true_sum
    accuracy = tp_fp_sum / len(prediction)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision*recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    a = (prediction == ground_truth)
    return sum(a) / len(prediction)
