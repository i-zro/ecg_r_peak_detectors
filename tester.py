from multiprocessing import Pool
from timer import timer
import wfdb
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import detectors
import sample_time_converter


# R_SYMBOLS = ['N', 'V', 'L', 'R', '/', 'f', 'A', 'E', 'Q', 'F', 'j', 'J', 'a', 'S', 'e']
R_SYMBOLS = ['N', 'V', 'L', 'R', '/', 'f', 'A', 'E', 'Q', 'F', 'j', 'J', 'a', 'S', 'e', 'r', 'F', 'n', '?']
DETECTION_X_RANGE = 58 # or 56
FILES = list({f.split('.')[0] for f in listdir('./db') if isfile(join('./db', f))} - {'.'})

TESTED_ALG = 1

ALG1 = detectors.alg1_spanish
ALG1_FILE_OUT = 'results/alg1_spanish_per_file.json'
ALG2 = detectors.alg2_chinese
ALG2_FILE_OUT = 'results/alg2_chinese_per_file.json'
ALG3 = detectors.alg3_iranian_test
ALG3_FILE_OUT = 'results/alg3_iranian_per_file.json'
ALG4 = detectors.alg4_polish
ALG4_FILE_OUT = 'results/alg4_polish_per_file.json'
ALG5 = detectors.alg5_pan_tompkins
ALG5_FILE_OUT = 'results/alg5_pan_tompkins_per_file.json'

ALG_FILE_DICT = {1: (ALG1, ALG1_FILE_OUT),
                 2: (ALG2, ALG2_FILE_OUT),
                 3: (ALG3, ALG3_FILE_OUT),
                 4: (ALG4, ALG4_FILE_OUT),
                 5: (ALG5, ALG5_FILE_OUT)}


@timer
def test_all_multi_thr():
    res = []
    with Pool(16) as p:
        res = p.map(test_file, FILES)
    return res


@timer
def test_all_single_thr():
    stats = []
    for file in FILES:
        stats.append(test_file(file))
    return stats


def test_file(file):
    filename = 'db/' + file
    record = wfdb.rdrecord(filename)
    signal = list(map(lambda x: x[0], record.p_signal))
    found_r_peaks = ALG_FILE_DICT[TESTED_ALG][0](signal)
    r_peaks_annotated = get_r_peaks_from(wfdb.rdann(filename, 'atr'))

    t_pos, f_pos, f_neg = binary_classifier(r_peaks_annotated, found_r_peaks)
    return create_stats(file, len(r_peaks_annotated), t_pos, f_pos, f_neg)


def get_r_peaks_from(ann):
    return list(map(lambda x: x[0], filter(lambda x: x[1] in R_SYMBOLS, zip(ann.sample, ann.symbol))))


def binary_classifier(r_peaks_annotated, found_r_peaks):
    t_pos = 0
    if len(r_peaks_annotated) > 0:
        bitmap_len = r_peaks_annotated[-1] + DETECTION_X_RANGE
        found_bitmap = np.zeros(bitmap_len)
        for x in found_r_peaks:
            if x < bitmap_len:
                found_bitmap[x] = 1
        for x in r_peaks_annotated:
            left_thres = max(0, x - DETECTION_X_RANGE)
            right_thres = min(bitmap_len, x + DETECTION_X_RANGE + 1)
            found = False
            for i in range(left_thres, right_thres):
                if found_bitmap[i]:
                    found = True
                    found_bitmap[i] = 0
                    break
            if found:
                t_pos += 1
            # else:
            #     print('x: ' + str(x) + ', time: ' +  sample_time_converter.sample_to_time(x))
    else:
        t_pos = 0
    f_neg = len(r_peaks_annotated) - t_pos
    f_pos = len(found_r_peaks) - t_pos
    print('t_pos:', t_pos, 'f_pos:', f_pos, 'f_neg: ', f_neg)
    return t_pos, f_pos, f_neg


def create_stats(filename, annotation_count, t_pos, f_pos, f_neg):
    return {
        'tape': filename,
        'TB': annotation_count,
        'TP': t_pos,
        'FN': f_neg,
        'FP': f_pos
    }


if __name__ == '__main__':
    res = test_all_multi_thr()

    f = open(ALG_FILE_DICT[TESTED_ALG][1], 'w')
    f.write(json.dumps(res))
    f.close()
