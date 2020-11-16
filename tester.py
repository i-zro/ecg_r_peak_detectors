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


@timer
def test_all_multi_thr():
    res = []
    with Pool(16) as p:
        res = p.map(test_file, FILES)
    return [itm for sublist in res for itm in sublist]  # flatten


@timer
def test_all_single_thr():
    stats = []
    for file in FILES:
        stats.append(test_file(file))
    return stats


def test_file(file):
    filename = 'db/' + file
    record = wfdb.rdrecord(filename)
    record_signal_ch0 = list(map(lambda x: x[0], record.p_signal))

    anno = wfdb.rdann(filename, 'atr')
    anno_r_peaks = get_r_peaks_from(anno)
    anno_r_peaks_x = list(map(lambda x: x[0], anno_r_peaks))
    annotation_count = len(anno_r_peaks)

    file_stats = []
    t_pos, f_pos, f_neg = test_detection(record_signal_ch0, anno_r_peaks_x)
    file_stats.append(create_stats(file, annotation_count, t_pos, f_pos, f_neg))

    return file_stats


def get_r_peaks_from(ann):
    return list(filter(lambda x: x[1] in R_SYMBOLS, zip(ann.sample, ann.symbol)))


def test_detection(record_signal_ch0, anno_r_peaks_x):
    found_r_peaks = detectors.ff_my(record_signal_ch0)
    t_pos, f_pos, f_neg = calculate_stats_for_tests_bitmap(anno_r_peaks_x, found_r_peaks)
    return t_pos, f_pos, f_neg


# @timer
def calculate_stats_for_tests_bitmap(anno_r_peaks_x, found_r_peaks_x):
    return calculate_stats_for_tests_bitmap2(anno_r_peaks_x, found_r_peaks_x)


def calculate_stats_for_tests_bitmap2(annotated_x, detected_x):
    t_pos = 0
    print('annotated:', len(annotated_x), '/ detected:', len(detected_x))

    if len(annotated_x) > 0:
        bitmap_len = annotated_x[len(annotated_x) - 1] + DETECTION_X_RANGE
        anno_bitmap = np.zeros(bitmap_len)
        det_bitmap = np.zeros(bitmap_len)
        for x in annotated_x:
            for i in range(0, DETECTION_X_RANGE):
                if x - i > 0:
                    anno_bitmap[x - i] = 1
                anno_bitmap[x + i] = 1
        for x in detected_x:
            if x < bitmap_len:
                det_bitmap[x] = 1

        t_pos = 0
        f_pos = 0
        for i in range(0, bitmap_len):
            if det_bitmap[i]:
                if anno_bitmap[i]:
                    t_pos += 1
                else:
                    f_pos += 1

        f_neg = max(0, len(annotated_x) - t_pos)
    else:
        t_pos = 0
        f_neg = 0
        f_pos = len(detected_x)
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

    f = open('tmp.json', 'w')
    f.write(json.dumps(res))
    f.close()
