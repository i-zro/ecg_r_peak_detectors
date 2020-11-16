import json
import csv

FILE_IN = 'tmp.json'
PRINT_SUMARY = True
SAVE_TO_FILE = True
FILE_OUT = 'results.csv'


def main():
    annotations = 0
    tp = 0
    fp = 0
    fn = 0
    with open(FILE_IN) as json_file:
        data = json.load(json_file)
        for elem in data:
            annotations += elem['annotation_count']
            tp += elem['true_positive']
            fp += elem['false_positive']
            fn += elem['false_negative']
    if SAVE_TO_FILE:
        with open(FILE_OUT, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Total peaks', annotations])
            writer.writerow(['TP', tp])
            writer.writerow(['FP', fp])
            writer.writerow(['FN', fn])
    if PRINT_SUMARY:
        print('Total peaks:\t\t' + str(annotations))
        print('True positives:\t\t' + str(tp))
        print('False positives:\t' + str(fp))
        print('False negatives:\t' + str(fn))


if __name__ == '__main__':
    main()
