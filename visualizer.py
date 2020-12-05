import matplotlib.pyplot as plt
import wfdb
import numpy as np
import tester
import detectors
import sample_time_converter as conv

FILE_NUMBER = '217'
FILEPATH = 'db/' + FILE_NUMBER
HZ = 360
REF_SAMPLES = 40
X_TO_Y_RATIO = 144 # 1 unit in ecg square ratio: 200ms / 0.5mv = 72samples / 0.5 = 144
TIME_STEP_PER_ROW = 10
ROW_LENGTH = TIME_STEP_PER_ROW * HZ # 10s, 360Hz
ROWS_PER_IMAGE = 3
IMAGE_LENGTH = ROWS_PER_IMAGE * ROW_LENGTH
IMAGES_NUM = int(30 * 60 * HZ / IMAGE_LENGTH) # == 60, = 30minutes * 60seconds * 360Hz / image length


def get_r_samples(ann):
    return list(filter(lambda x: x[1] in tester.R_SYMBOLS, zip(ann.sample, ann.symbol)))


def get_plot_data():
    record = wfdb.rdrecord(FILEPATH)
    signal_ch0 = list(map(lambda x: x[0], record.p_signal))
    ecg = np.array(signal_ch0)
    peaks_r = detectors.alg1_spanish(ecg)
    # peaks_r = detectors.alg4_polish(ecg)

    ann = wfdb.rdann(FILEPATH, 'atr')
    annotations = get_r_samples(ann)
    anno_peaks_x = list(map(lambda x: x[0], annotations))

    return signal_ch0, anno_peaks_x, peaks_r


def plot_data():
    signal_ch0, anno_peaks_x, peaks_r = get_plot_data()
    TP, FP, FN = binary_classifier(anno_peaks_x, peaks_r)
    time_start = 0
    for i in range(0, IMAGES_NUM):
        fig = plt.figure()
        for j in range(0, ROWS_PER_IMAGE):
            ax = fig.add_subplot(ROWS_PER_IMAGE * 100 + 10 + j + 1) # e.g. 311, 312, 313
            ax.set_aspect(X_TO_Y_RATIO)
            samples = list(range(time_start * HZ, time_start * HZ + ROW_LENGTH))
            ax.plot(samples, signal_ch0[samples[0] : (samples[-1] + 1)])
            for sample in samples:
                if sample in TP:
                    ax.plot(sample, signal_ch0[sample], 'go')
                if sample in FP:
                    ax.plot(sample, signal_ch0[sample], 'ro')
                if sample in FN:
                    ax.plot(sample, signal_ch0[sample], 'rx')
            ax.title.set_text('FROM: ' + conv.sample_to_time(samples[0]) + '    TO: ' + conv.sample_to_time(samples[-1]))
            plt.savefig('images/' + FILE_NUMBER + '_' + str(i) + '.png')
            time_start += TIME_STEP_PER_ROW
        plt.close(fig)


def binary_classifier(r_peaks_annotated, found_r_peaks):
    TP = []
    FN = []
    FP = []
    t_pos = 0
    bitmap_len = r_peaks_annotated[-1] + tester.DETECTION_X_RANGE
    found_bitmap = np.zeros(bitmap_len)
    for x in found_r_peaks:
        if x < bitmap_len:
            found_bitmap[x] = 1
    if len(r_peaks_annotated) > 0:
        for x in r_peaks_annotated:
            left_thres = max(0, x - tester.DETECTION_X_RANGE)
            right_thres = min(bitmap_len, x + tester.DETECTION_X_RANGE + 1)
            found = False
            for i in range(left_thres, right_thres):
                if found_bitmap[i]:
                    found = True
                    TP.append(i)
                    found_bitmap[i] = 0
                    break
            if found:
                t_pos += 1
            else:
                FN.append(x)
    else:
        t_pos = 0
    for i in range(0, bitmap_len):
        if found_bitmap[i]:
            FP.append(i)
    f_neg = len(r_peaks_annotated) - t_pos
    f_pos = len(found_r_peaks) - t_pos
    print('t_pos:', t_pos, 'f_pos:', f_pos, 'f_neg: ', f_neg)
    print('TP:', len(TP), 'FP:', len(FP), 'FN: ', len(FN))
    return TP, FP, FN


if __name__ == '__main__':
    plot_data()
