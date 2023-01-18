import numpy as np


def getpredict(result, T1 = 0.5, T2 = 0.4):
    for i in range(len(result)):
        r = []
        maxl, maxj = -1, -1
        for j in range(len(result[i])):
            if result[i][j] > T1:
                r += [j]
            if result[i][j] > maxl:
                maxl = result[i][j]
                maxj = j
        if len(r) == 0:
            if maxl <= T2:
                r = [36]
            else:
                r += [maxj]
        result[i] = r
    return result

def evaluate(preds, labels):
    '''
        preds: list (N, C)
        labels: list (N, C)
    '''


    preds = getpredict(preds) # (N, num_predicted_labels)
    labels = getpredict(labels) # (N, num_correct_labels)
    correct_sys, all_sys = 0, 0
    correct_gt = 0
    
    for i in range(len(labels)):
        for id in labels[i]:
            if id != 36:
                correct_gt += 1
                if id in preds[i]:
                    correct_sys += 1
        for id in preds[i]:
            if id != 36:
                all_sys += 1

    precision = correct_sys/all_sys if all_sys != 0 else 1
    recall = correct_sys/correct_gt if correct_gt != 0 else 0
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1, preds, labels


def evaluate_f1c(devp, data):
    index = 0
    precisions = []
    recalls = []
    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            correct_sys, all_sys = 0, 0
            correct_gt = 0
            
            x = data[i][1][j]["x"].lower().strip()
            y = data[i][1][j]["y"].lower().strip()
            t = {}
            for k in range(len(data[i][1][j]["rid"])):
                if data[i][1][j]["rid"][k] != 36:
                    t[data[i][1][j]["rid"][k]] = data[i][1][j]["t"][k].lower().strip()

            l = set(data[i][1][j]["rid"]) - set([36])

            ex, ey = False, False
            et = {}
            for r in range(36):
                et[r] = r not in l

            for k in range(len(data[i][0])):
                o = set(devp[index]) - set([36])
                e = set()
                if x in data[i][0][k].lower():
                    ex = True
                if y in data[i][0][k].lower():
                    ey = True
                if k == len(data[i][0])-1:
                    ex = ey = True
                    for r in range(36):
                        et[r] = True
                for r in range(36):
                    if r in t:
                        if t[r] != "" and t[r] in data[i][0][k].lower():
                            et[r] = True
                    if ex and ey and et[r]:
                        e.add(r)
                correct_sys += len(o & l & e)
                all_sys += len(o & e)
                correct_gt += len(l & e)
                index += 1

            precisions += [correct_sys/all_sys if all_sys != 0 else 1]
            recalls += [correct_sys/correct_gt if correct_gt != 0 else 0]

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1