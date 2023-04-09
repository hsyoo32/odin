from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, roc_curve, auc
import numpy as np
import utility as util
from sklearn.linear_model import LogisticRegression

def compute_accuracy_using_classifier(train_edges, test_edges, train_edges_uncon, test_edges_uncon, embeddings, emb_algo, 
    result_filename):

    flag = 0
    X_train = []; Y_train = []  
    for edge in train_edges + train_edges_uncon:
        flag += 1
        if flag > len(train_edges):
            y = 0
        else:
            y = 1

        if len(embeddings) < 2:
            emb1 = np.array(embeddings[0][edge[0]])
            emb2 = np.array(embeddings[0][edge[1]])
        else:
            emb1 = np.array(embeddings[0][edge[0]])
            emb2 = np.array(embeddings[1][edge[1]])
        edge_emb = util.aggregate_edge_emb(emb1, emb2)
        X_train.append(edge_emb)
        Y_train.append(y)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    flag = 0
    X_test = []; Y_test = []
    for edge in test_edges + test_edges_uncon:
        flag += 1
        if flag > len(test_edges):
            y = 0
        else:
            y = 1

        if len(embeddings) < 2:
            emb1 = np.array(embeddings[0][edge[0]])
            emb2 = np.array(embeddings[0][edge[1]])
        else:
            emb1 = np.array(embeddings[0][edge[0]])
            emb2 = np.array(embeddings[1][edge[1]])
        edge_emb = util.aggregate_edge_emb(emb1, emb2)
        X_test.append(edge_emb)   
        Y_test.append(y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    #train LR and test
    lr = LogisticRegression(max_iter=10000)  # max_iter=10000
    lr.fit(X_train, Y_train)
    test_y_score = lr.predict_proba(X_test)[:, 1]
    test_y_pred = lr.predict(X_test)

    lp_auc_score_macro = roc_auc_score(Y_test, test_y_score, average='macro')
    lp_f1_score_macro = f1_score(Y_test, test_y_pred, average='macro')
    lp_f1_score_micro = f1_score(Y_test, test_y_pred, average='micro')
    lp_f1_score_binary = f1_score(Y_test, test_y_pred, average='binary')
    ap_score = average_precision_score(Y_test, test_y_pred)
    lp_acc = accuracy_score(Y_test, test_y_pred)

    #print("final reuslt filename: {}".format(result_filename))
    with open(result_filename, "w+") as f:
        f.writelines("auc_score_macro\t" + str(lp_auc_score_macro) + "\n")
        f.writelines("f1_score_macro\t" + str(lp_f1_score_macro) + "\n")
        f.writelines("f1_score_micro\t" + str(lp_f1_score_micro) + "\n")
        f.writelines("f1_score_binary\t" + str(lp_f1_score_binary) + "\n")
        f.writelines("ap\t" + str(ap_score) + "\n")
        f.writelines("acc\t" + str(lp_acc))