import torch
import torch.nn as nn
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import tqdm
import copy
from sklearn import metrics 
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from mlClassifier.logging import logger

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(15, 15)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(15, 15)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(15, 15)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

def nn_train(device, model, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
    model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_epochs = epochs   # number of epochs to run
    batch_size = batch_size  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights_for_acc = None
    f1_for_best_acc, precision_for_best_acc, recall_for_best_acc = None, None, None

    best_f1 = - np.inf   # init to negative infinity
    best_weights_for_f1 = None
    acc_for_best_f1, precision_for_best_f1, recall_for_best_f1 = None, None, None

    best_precision = - np.inf   # init to negative infinity
    best_weights_for_precision = None
    acc_for_best_precision, f1_for_best_precision, recall_for_best_precision = None, None, None

    best_recall = - np.inf   # init to negative infinity
    best_weights_for_recall = None
    acc_for_best_recall, f1_for_best_recall, precision_for_best_recall = None, None, None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                X_batch = X_batch.to(device)
                y_batch = y_train[start:start+batch_size]
                y_batch = y_batch.to(device)
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        y_real = y_val.cpu().numpy()
        y_try = y_pred.round().detach().cpu().numpy()
        # print(metrics.accuracy_score(y_real, y_try), metrics.f1_score(y_real, y_try), metrics.precision_score(y_real, y_try), metrics.recall_score(y_real, y_try))
        # acc = (y_pred.round() == y_val).float().mean()
        # acc = float(acc)
        acc = metrics.accuracy_score(y_real, y_try)
        f1 = metrics.f1_score(y_real, y_try)
        precision = metrics.precision_score(y_real, y_try)
        recall = metrics.recall_score(y_real, y_try)
        
        #print(acc)
        if acc > best_acc:
            best_acc = acc
            f1_for_best_acc, precision_for_best_acc, recall_for_best_acc = f1, precision, recall
            best_weights_for_acc = copy.deepcopy(model.state_dict())

        if f1 > best_f1:
            best_f1 = f1
            acc_for_best_f1, precision_for_best_f1, recall_for_best_f1 = acc, precision, recall
            best_weights_for_f1 = copy.deepcopy(model.state_dict())

        if precision > best_precision:
            best_precision = precision
            acc_for_best_precision, f1_for_best_precision, recall_for_best_precision = acc, f1, recall
            best_weights_for_precision = copy.deepcopy(model.state_dict())

        if recall > best_recall:
            best_recall = recall
            acc_for_best_recall, f1_for_best_recall, precision_for_best_recall = acc, f1, precision
            best_weights_for_recall = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    # model.load_state_dict(best_weights)
    return [[best_acc, f1_for_best_acc, precision_for_best_acc, recall_for_best_acc], 
            [best_f1, acc_for_best_f1, precision_for_best_f1, recall_for_best_f1],
            [best_precision, acc_for_best_precision, f1_for_best_precision, recall_for_best_precision],
            [best_recall, acc_for_best_recall, f1_for_best_recall, precision_for_best_recall]], best_weights_for_acc, best_weights_for_f1, best_weights_for_precision, best_weights_for_recall

def train_Grid_CV_RandomForestClassifier(params, X, y):
    best_acc, best_f1, best_precision, best_recall = -np.inf, -np.inf, -np.inf, -np.inf
    best_model_for_acc, best_model_for_f1, best_model_for_precision, best_model_for_recall = None, None, None, None
    for n_estimators in params.n_estimators:
        for criterion in params.criterion:
            for max_depth in params.max_depth:
                clf = RandomForestClassifier(n_estimators = n_estimators, criterion=criterion, max_depth=max_depth)  
                np.random.seed(0)
                cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
                acc, f1, precision, recall = 0, 0, 0, 0
                for (train, test), i in zip(cv.split(X, y), range(10)):
                    clf.fit(X.iloc[train], y.iloc[train])
                    y_pred = clf.predict(X.iloc[test])
                    y_test = y.iloc[test]
                    acc, f1, precision, recall = metrics.accuracy_score(y_test, y_pred) + acc, metrics.f1_score(y_test, y_pred) + f1, metrics.precision_score(y_test, y_pred) + precision, metrics.recall_score(y_test, y_pred) + recall
                acc, f1, precision, recall = acc/10, f1/10, precision/10, recall/10
                if acc > best_acc:
                    best_acc = acc
                    best_model_for_acc = clf
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_for_f1 = clf
                if precision > best_precision:
                    best_precision = precision
                    best_model_for_precision = clf
                if recall > best_recall:
                    best_recall = recall
                    best_model_for_recall = clf
                logger.info(f"Finished Cross Validation for the following hyperparameters | Number of Estimators: {n_estimators}, Criterion: {criterion}, Max Depth: {max_depth}")
    return {"best_acc": (best_acc, best_model_for_acc),
            "best_f1": (best_f1, best_model_for_f1),
            "best_precision": (best_recall, best_model_for_precision),
            "best_recall": (best_precision, best_model_for_recall)}

def train_Grid_CV_XGBClassifier(params, X, y):
    best_acc, best_f1, best_precision, best_recall = -np.inf, -np.inf, -np.inf, -np.inf
    best_model_for_acc, best_model_for_f1, best_model_for_precision, best_model_for_recall = None, None, None, None
    for n_estimators in params.n_estimators:
        for learning_rate in params.learning_rate:
            for max_depth in params.max_depth:
                clf = XGBClassifier(n_estimators = n_estimators, learning_rate=learning_rate, max_depth=max_depth)  
                np.random.seed(0)
                cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
                acc, f1, precision, recall = 0, 0, 0, 0
                for (train, test), i in zip(cv.split(X, y), range(10)):
                    clf.fit(X.iloc[train], y.iloc[train])
                    y_pred = clf.predict(X.iloc[test])
                    y_test = y.iloc[test]
                    acc, f1, precision, recall = metrics.accuracy_score(y_test, y_pred) + acc, metrics.f1_score(y_test, y_pred) + f1, metrics.precision_score(y_test, y_pred) + precision, metrics.recall_score(y_test, y_pred) + recall
                acc, f1, precision, recall = acc/10, f1/10, precision/10, recall/10
                if acc > best_acc:
                    best_acc = acc
                    best_model_for_acc = clf
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_for_f1 = clf
                if precision > best_precision:
                    best_precision = precision
                    best_model_for_precision = clf
                if recall > best_recall:
                    best_recall = recall
                    best_model_for_recall = clf
                logger.info(f"Finished Cross Validation for the following hyperparameters | Number of Estimators: {n_estimators}, Learning Rate: {learning_rate}, Max Depth: {max_depth}")
    return {"best_acc": (best_acc, best_model_for_acc),
            "best_f1": (best_f1, best_model_for_f1),
            "best_precision": (best_recall, best_model_for_precision),
            "best_recall": (best_precision, best_model_for_recall)}

def train_Grid_CV_CatBoostClassifier(params, X, y):
    best_acc, best_f1, best_precision, best_recall = -np.inf, -np.inf, -np.inf, -np.inf
    best_model_for_acc, best_model_for_f1, best_model_for_precision, best_model_for_recall = None, None, None, None
    for iterations in params.iterations:
        for learning_rate in params.learning_rate:
            for depth in params.depth:
                clf = CatBoostClassifier(iterations = iterations, learning_rate=learning_rate, depth=depth, verbose=False)  
                np.random.seed(0)
                cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
                acc, f1, precision, recall = 0, 0, 0, 0
                for (train, test), i in zip(cv.split(X, y), range(10)):
                    clf.fit(X.iloc[train], y.iloc[train])
                    y_pred = clf.predict(X.iloc[test])
                    y_test = y.iloc[test]
                    acc, f1, precision, recall = metrics.accuracy_score(y_test, y_pred) + acc, metrics.f1_score(y_test, y_pred) + f1, metrics.precision_score(y_test, y_pred) + precision, metrics.recall_score(y_test, y_pred) + recall
                acc, f1, precision, recall = acc/10, f1/10, precision/10, recall/10
                if acc > best_acc:
                    best_acc = acc
                    best_model_for_acc = clf
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_for_f1 = clf
                if precision > best_precision:
                    best_precision = precision
                    best_model_for_precision = clf
                if recall > best_recall:
                    best_recall = recall
                    best_model_for_recall = clf
                logger.info(f"Finished Cross Validation for the following hyperparameters | Iterations: {iterations}, Learning Rate: {learning_rate}, Depth: {depth}")
    return {"best_acc": (best_acc, best_model_for_acc),
            "best_f1": (best_f1, best_model_for_f1),
            "best_precision": (best_recall, best_model_for_precision),
            "best_recall": (best_precision, best_model_for_recall)}

def train_Grid_CV_AdaBoostClassifier(params, X, y):
    best_acc, best_f1, best_precision, best_recall = -np.inf, -np.inf, -np.inf, -np.inf
    best_model_for_acc, best_model_for_f1, best_model_for_precision, best_model_for_recall = None, None, None, None
    for n_estimators in params.n_estimators:
        for learning_rate in params.learning_rate:
            clf = AdaBoostClassifier(n_estimators = n_estimators, learning_rate=learning_rate)  
            np.random.seed(0)
            cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
            acc, f1, precision, recall = 0, 0, 0, 0
            for (train, test), i in zip(cv.split(X, y), range(10)):
                clf.fit(X.iloc[train], y.iloc[train])
                y_pred = clf.predict(X.iloc[test])
                y_test = y.iloc[test]
                acc, f1, precision, recall = metrics.accuracy_score(y_test, y_pred) + acc, metrics.f1_score(y_test, y_pred) + f1, metrics.precision_score(y_test, y_pred) + precision, metrics.recall_score(y_test, y_pred) + recall
            acc, f1, precision, recall = acc/10, f1/10, precision/10, recall/10
            if acc > best_acc:
                best_acc = acc
                best_model_for_acc = clf
            if f1 > best_f1:
                best_f1 = f1
                best_model_for_f1 = clf
            if precision > best_precision:
                best_precision = precision
                best_model_for_precision = clf
            if recall > best_recall:
                best_recall = recall
                best_model_for_recall = clf
            logger.info(f"Finished Cross Validation for the following hyperparameters | Number of Estimators: {n_estimators}, Learning Rare: {learning_rate}")
    return {"best_acc": (best_acc, best_model_for_acc),
            "best_f1": (best_f1, best_model_for_f1),
            "best_precision": (best_recall, best_model_for_precision),
            "best_recall": (best_precision, best_model_for_recall)}

def train_Grid_CV_Neural_Network_Classifier(params, X_tensor, y_tensor):
    best_weights_for_acc, best_weights_for_f1, best_weights_for_precision, best_weights_for_recall = None, None, None, None
    best_avg_acc, best_avg_f1, best_avg_precision, best_avg_recall = - np.inf, - np.inf, - np.inf, - np.inf
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    for epochs in params.epochs:
        for lr in params.lrs:
            for batch_size in params.batch_sizes:
                warnings.filterwarnings('ignore')
                # define 5-fold cross validation test harness
                np.random.seed(0)
                kfold = StratifiedKFold(n_splits=10, shuffle=True)
                #cv_scores = []
                sum_results = np.zeros((4, 4))
                for train, test in kfold.split(X_tensor, y_tensor):
                    torch.manual_seed(0)
                    model = Deep().apply(weights_init)
                    results, weights_for_acc, weights_for_f1, weights_for_precision, weights_for_recall = nn_train(device, model, X_tensor[train], y_tensor[train], X_tensor[test], y_tensor[test], epochs, lr, batch_size)
                    sum_results = np.add(sum_results, np.array(results))
                sum_results = sum_results/10
                if sum_results[0][0] > best_avg_acc:
                    best_avg_acc = sum_results[0][0]
                    best_weights_for_acc = weights_for_acc
                if sum_results[1][0] > best_avg_f1:
                    best_avg_f1 = sum_results[1][0]
                    best_weights_for_f1 = weights_for_f1
                if sum_results[2][0] > best_avg_precision:
                    best_avg_precision = sum_results[2][0]
                    best_weights_for_precision = weights_for_precision
                if sum_results[3][0] > best_avg_recall:
                    best_avg_recall = sum_results[3][0]
                    best_weights_for_recall = weights_for_recall
                logger.info(f"Finished Cross Validation for the following hyperparameters | Epochs: {epochs}, Learning Rate: {lr}, Batch Size: {batch_size}")
    return {"best_acc": (best_avg_acc, best_weights_for_acc),
            "best_f1": (best_avg_f1, best_weights_for_f1),
            "best_precision": (best_avg_recall, best_weights_for_precision),
            "best_recall": (best_avg_precision, best_weights_for_recall)}