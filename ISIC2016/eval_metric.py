import csv
import numpy as np

def avg_precision(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))
    # record muli class precision
    precision = []
    for cls in range(7):
        correct = 0
        cnt = 0
        for i in range(len(pred)):
            if pred[i] == cls:
                cnt += 1
                if pred[i] == gt[i]:
                    correct += 1
        precision.append(correct / cnt)
    return precision

def avg_recall(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))
    # record muli class recall
    recall = []
    for cls in range(7):
        correct = 0
        cnt = 0
        for i in range(len(gt)):
            if gt[i] == cls:
                cnt += 1
                if pred[i] == gt[i]:
                    correct += 1
        recall.append(correct / cnt)
    return recall

if __name__ == "__main__":
    result_file = 'noattn_ce.csv'
    precision = avg_precision(result_file)
    recall = avg_recall(result_file)
    print("precision")
    print(precision)
    print(np.mean(np.array(precision)))
    print("\nrecall")
    print(recall)
    print(np.mean(np.array(recall)))
