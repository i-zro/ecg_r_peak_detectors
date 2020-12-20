import json
import csv

PRINT_SUMMARY = True
SAVE_TO_FILE = True

TESTED_ALG = 1

ALG1_FILE_IN = 'results/alg1_spanish_per_file.json'
ALG1_FILE_OUT = 'results/alg1_spanish_summary.csv'
ALG2_FILE_IN = 'results/alg2_chinese_per_file.json'
ALG2_FILE_OUT = 'results/alg2_chinese_summary.csv'
ALG3_FILE_IN = 'results/alg3_iranian_per_file.json'
ALG3_FILE_OUT = 'results/alg3_iranian_summary.csv'
ALG4_FILE_IN = 'results/alg4_polish_per_file.json'
ALG4_FILE_OUT = 'results/alg4_polish_summary.csv'
ALG5_FILE_IN = 'results/alg5_pan_tompkins_per_file.json'
ALG5_FILE_OUT = 'results/alg5_pan_tompkins_summary.csv'

ALG_FILE_DICT = {1: (ALG1_FILE_IN, ALG1_FILE_OUT),
                 2: (ALG2_FILE_IN, ALG2_FILE_OUT),
                 3: (ALG3_FILE_IN, ALG3_FILE_OUT),
                 4: (ALG4_FILE_IN, ALG4_FILE_OUT),
                 5: (ALG5_FILE_IN, ALG5_FILE_OUT)}


def main():
    annotations = 0
    tp = 0
    fp = 0
    fn = 0
    rows = []
    with open(ALG_FILE_DICT[TESTED_ALG][0]) as json_file:
        data = json.load(json_file)
        for elem in data:
            se = elem['TP'] / (elem['TP'] + elem['FN']) * 100
            p = elem['TP'] / (elem['TP'] + elem['FP']) * 100
            der = (elem['FP'] + elem['FN']) / elem['TB'] * 100
            rows.append([elem['tape'], elem['TB'], elem['FN'],
                         elem['FP'], "{:.2f}".format(se), "{:.2f}".format(p), "{:.2f}".format(der)])
            annotations += elem['TB']
            tp += elem['TP']
            fp += elem['FP']
            fn += elem['FN']
    rows.sort(key=lambda x: x[0])
    se = tp / (tp + fn) * 100
    p = tp / (tp + fp) * 100
    der = (fp + fn) / annotations * 100
    rows.append(['TOTAL', annotations, fn, fp, "{:.2f}".format(se), "{:.2f}".format(p), "{:.2f}".format(der)])
    if SAVE_TO_FILE:
        with open(ALG_FILE_DICT[TESTED_ALG][1], 'w', newline='') as csvfile:
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
