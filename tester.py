from multiprocessing import Pool
from timer import timer
import wfdb
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import detectors


R_SYMBOLS = ['N', 'V', 'L', 'R', '/', 'f', 'A', 'E', 'Q', 'F', 'j', 'J', 'a', 'S', 'e']
DETECTION_X_RANGE = 58
FILES = list({f.split('.')[0] for f in listdir('./db') if isfile(join('./db', f))} - {'.'})
FILE_OUT = 'results_per_file.json'


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
    found_r_peaks = detectors.ff_my(signal)
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
    else:
        t_pos = 0
    f_neg = len(r_peaks_annotated) - t_pos
    f_pos = len(found_r_peaks) - t_pos
    print('t_pos:', t_pos, 'f_pos:', f_pos, 'f_neg: ', f_neg)
    return t_pos, f_pos, f_neg


def create_stats(filename, annotation_count, t_pos, f_pos, f_neg):
    return {
        'filename': filename,
        'annotation_count': annotation_count,
        'true_positive': t_pos,
        'false_positive': f_pos,
        'false_negative': f_neg
    }


if __name__ == '__main__':
    res = test_all_multi_thr()

    f = open(FILE_OUT, 'w')
    f.write(json.dumps(res))
    f.close()
