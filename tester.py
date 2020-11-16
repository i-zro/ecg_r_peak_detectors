from multiprocessing import Pool
from timer import timer
import wfdb
import numpy as np
import detectors
import json
from os import listdir
from os.path import isfile, join

FILES_TO_SKIP = []
SAMPLES_PER_SECOND = 360

FILES = list({f.split('.')[0] for f in listdir('./db') if isfile(join('./db', f))} - set(FILES_TO_SKIP) - {'.'})

DETECTION_X_RANGE2 = 58
R_SYMBOLS = ['N', 'V', 'L', 'R', '/', 'f', 'A', 'E', 'Q', 'F', 'j', 'J', 'a', 'S', 'e']


def get_r_samples(ann):
    return list(filter(lambda x: x[1] in R_SYMBOLS, zip(ann.sample, ann.symbol)))

def get_r_peaks_from(anno):
    return get_r_samples(anno)


def calculate_stats_for_tests_bitmap2(annotated_x, detected_x):
    t_pos = 0
    print('annotated:', len(annotated_x), '/ detected:', len(detected_x))

    if len(annotated_x) > 0:
        bitmap_len = annotated_x[len(annotated_x) - 1] + DETECTION_X_RANGE2
        anno_bitmap = np.zeros(bitmap_len)
        det_bitmap = np.zeros(bitmap_len)
        for x in annotated_x:
            for i in range(0, DETECTION_X_RANGE2):
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


# @timer
def calculate_stats_for_tests_bitmap(anno_r_peaks_x, found_r_peaks_x):
    return calculate_stats_for_tests_bitmap2(anno_r_peaks_x, found_r_peaks_x)


def test_detection(file_number, record_signal_ch0, anno_r_peaks_x, ref_samples, threshold):
    found_r_peaks = detectors.ff_my(record_signal_ch0)
    found_r_peaks_x = list(map(lambda x: x[0], found_r_peaks))
    t_pos, f_pos, f_neg = calculate_stats_for_tests_bitmap(anno_r_peaks_x, found_r_peaks_x)
    return t_pos, f_pos, f_neg


def is_in_some_fragment(x, ranges):
    for range in ranges:
        start_sample = convert_time_to_sample(range[0])
        end_sample = convert_time_to_sample(range[1])
        if start_sample <= x <= end_sample:
            return True
    return False


def is_out_of_fragment(x, ranges):
    for range in ranges:
        if range[0] <= x <= range[1]:
            return False
    return True


def sample_to_time(sample):
    return str(hour_of_sample(sample)).zfill(2) + ':' + \
           str(min_of_sample(sample)).zfill(2) + ':' + \
           str(sec_of_sample(sample)).zfill(2) + '.' + \
           str(msec_of_sample(sample)).zfill(3)


def hour_of_sample(sample):
    return int(sample / (SAMPLES_PER_SECOND * 3600))


def min_of_sample(sample):
    return int((sample % (SAMPLES_PER_SECOND * 3600)) / (SAMPLES_PER_SECOND * 60))


def sec_of_sample(sample):
    return int((sample % (SAMPLES_PER_SECOND * 60)) / (SAMPLES_PER_SECOND))


def msec_of_sample(sample):
    return int((sample % (SAMPLES_PER_SECOND)) / SAMPLES_PER_SECOND * 1000)


def convert_time_to_sample(time):
    hour = get_hour(time)
    min = get_min(time)
    sec = get_sec(time)
    msec = get_msec(time)
    return SAMPLES_PER_SECOND * (3600 * hour + 60 * min + sec + (msec / 1000))


def get_hour(time):
    return int(time[0:2])


def get_min(time):
    return int(time[3:5])


def get_sec(time):
    return int(time[6:8])


def get_msec(time):
    return int(time[9:12])


def create_stats(filename, annotaion_count, ref_samples, threshold, t_pos, f_pos, f_neg):
    return {
        'filename': filename,
        'annotation_count': annotaion_count,
        'ref_samples': ref_samples,
        'threshold': threshold,
        'true_positive': t_pos,
        'false_positive': f_pos,
        'false_negative': f_neg
    }


def test_file(file):
    filename = 'db/' + file
    record = wfdb.rdrecord(filename)
    record_signal_ch0 = list(map(lambda x: x[0], record.p_signal))

    anno = wfdb.rdann(filename, 'atr')
    anno_r_peaks = get_r_peaks_from(anno)
    anno_r_peaks_x = list(map(lambda x: x[0], anno_r_peaks))
    annotation_count = len(anno_r_peaks)

    file_stats = []
    t_pos, f_pos, f_neg = test_detection(file, record_signal_ch0, anno_r_peaks_x, 0, 0)
    file_stats.append(create_stats(file, annotation_count, 0, 0, t_pos, f_pos, f_neg))

    return file_stats


@timer
def test_all_single_thr():
    stats = []
    for file in FILES:
        stats.append(test_file(file))

    return stats


@timer
def test_all_multi_thr():
    res = []
    with Pool(16) as p:
        res = p.map(test_file, FILES)

    return [itm for sublist in res for itm in sublist]  # flatten


if __name__ == '__main__':
    res = test_all_multi_thr()

    f = open('tmp.json', 'w')
    f.write(json.dumps(res))
    f.close()
