from sklearn.metrics import label_ranking_loss

def evaluate_test_set(y_test,y_pred):
    return label_ranking_loss(y_test,y_pred)