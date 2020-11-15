import json


def main():
    annotations = 0
    tp = 0
    fp = 0
    fn = 0
    with open('tmp.json') as json_file:
        data = json.load(json_file)
        for elem in data:
            annotations += elem['annotation_count']
            tp += elem['true_positive']
            fp += elem['false_positive']
            fn += elem['false_negative']
    print('Total peaks:\t\t' + str(annotations))
    print('True positives:\t\t' + str(tp))
    print('False positives:\t' + str(fp))
    print('False negatives:\t' + str(fn))


if __name__ == '__main__':
    main()
