import json
import csv

PRINT_SUMMARY = True
SAVE_TO_FILE = True

# FILE_IN = 'results/alg1_spanish_per_file.json'
# FILE_OUT = 'results/alg1_spanish_summary.csv'
# FILE_IN = 'results/alg2_chinese_per_file.json'
# FILE_OUT = 'results/alg2_chinese_summary.csv'
# FILE_IN = 'results/alg3_iranian_per_file.json'
# FILE_OUT = 'results/alg3_iranian_summary.csv'
# FILE_IN = 'results/alg4_polish_per_file.json'
# FILE_OUT = 'results/alg4_polish_summary.csv'
FILE_IN = 'results/alg5_pan_tompkins_per_file.json'
FILE_OUT = 'results/alg5_pan_tompkins_summary.csv'


def main():
    annotations = 0
    tp = 0
    fp = 0
    fn = 0
    rows = []
    with open(FILE_IN) as json_file:
        data = json.load(json_file)
        for elem in data:
            se = elem['true_positive'] / (elem['true_positive'] + elem['false_negative']) * 100
            p = elem['true_positive'] / (elem['true_positive'] + elem['false_positive']) * 100
            der = (elem['false_positive'] + elem['false_negative']) / elem['annotation_count'] * 100
            rows.append([elem['filename'], elem['annotation_count'], elem['false_negative'],
                        elem['false_positive'], "{:.2f}".format(se), "{:.2f}".format(p), "{:.2f}".format(der)])
            annotations += elem['annotation_count']
            tp += elem['true_positive']
            fp += elem['false_positive']
            fn += elem['false_negative']
    rows.sort(key=lambda x: x[0])
    se = tp / (tp + fn) * 100
    p = tp / (tp + fp) * 100
    der = (fp + fn) / annotations * 100
    rows.append(['TOTAL', annotations, fn, fp, "{:.2f}".format(se), "{:.2f}".format(p), "{:.2f}".format(der)])
    if SAVE_TO_FILE:
        with open(FILE_OUT, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Tape', 'TB', 'FN', 'FP', 'Se(%)', '+P(%)', 'DER(%)'])
            for row in rows:
                writer.writerow(row)

    if PRINT_SUMMARY:
        print('Total peaks:\t\t' + str(annotations))
        print('True positives:\t\t' + str(tp))
        print('False positives:\t' + str(fp))
        print('False negatives:\t' + str(fn))


if __name__ == '__main__':
    main()
