import argparse
import os
import numpy as np
import torch
import pickle
import sys

from sklearn import metrics
from torch.autograd import Variable

from loader import load_data
from model import MRNet
from sklearn.linear_model import LogisticRegression

def get_parser():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()

        vol, label = batch
        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.data.cpu().numpy()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels

def evaluate(paths, task, use_gpu):

    model = MRNet()

    np.random.seed(42)

    all_preds = []

    for view in range(3):
        view_list = ['sagittal', 'coronal', 'axial']
        loader = load_data(paths, task, view_list[view], shuffle=False, use_gpu=use_gpu)

        state_dict = torch.load('src/models/'+view_list[view]+'-'+task, map_location=(None if use_gpu else 'cpu'))
        model.load_state_dict(state_dict)

        if use_gpu:
            model = model.cuda()

        loss, auc, preds, labels = run_model(model, loader)
        
        all_preds.append(preds)

        # print(f'{split} loss: {loss:0.4f}')
        # print(f'{split} AUC: {auc:0.4f}')

    preds = np.stack(all_preds, axis=1)
    return preds, labels

if __name__ == '__main__':
    # args = get_parser().parse_args()
    input_data_filename = sys.argv[1]
    output_prediction_filename = sys.argv[2]

    paths = [line.strip() for line in open(input_data_filename)]

    task_preds = []
    for task in ['abnormal', 'acl', 'meniscus']:
        test_X, test_y = evaluate(paths, task, torch.cuda.is_available()) #args.gpu)
        clf = pickle.load(open('src/models/lr-'+task, 'rb'))
        pred_y = clf.predict_proba(test_X)[:,1]
        task_preds.append(pred_y)


    task_preds = np.stack(task_preds, axis=1)
    np.savetxt(output_prediction_filename, task_preds, delimiter=',')

    '''
#    train_X, train_y = evaluate('train', args.task, args.gpu)

#    valid_X, valid_y = evaluate('valid', args.task, args.gpu)

    test_X, test_y = evaluate('test', args.task, args.gpu)

#    clf = LogisticRegression(class_weight='balanced')

#    clf.fit(train_X, train_y)

#    pickle.dump(clf, open('models/lr-'+args.task, 'wb'))

    clf = pickle.load(open('lr-'+args.task, 'rb'))
    pred_y = clf.predict_proba(test_X)[:,1]

#    predictions = np.load('../testset/predictions.npy')
#    fpr, tpr, threshold = metrics.roc_curve(test_y, predictions[:,2])
    fpr, tpr, threshold = metrics.roc_curve(test_y, pred_y)
    auc = metrics.auc(fpr, tpr)

    print(auc)
'''
