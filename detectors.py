import numpy as np
import math


def ff_basic_project(ecg):
    # Running median elements
    N = 8
    Nd = 7

    # Preprocessing variables
    padding = max(N, Nd, 10)
    i = 0
    current_window = []
    derivated_window = []

    # Threshold stage constants
    qrs_interval = 21 # 60ms
    rr_min = 72 # 200ms
    p_threshold = 0.7 * 360 / 128 + 4.7

    # Threshold stage variables
    counter = 0
    state = 1
    max_peak = (-1, -1, -1) # (index, value, shift)
    r_peaks = []
    r_peaks_pos = []
    th = 0

    for value in ecg:

        # collecting initial data
        if i < padding:
            current_window.append(value)
            derivated_window.append(value)
            i += 1
            continue

        # Preprocessing
        current_window.append(value)
        derivated_window.append(value - current_window[-Nd])

        integrated_value = np.sum(derivated_window[-1:-1-N:-1]) * 1 / (N - 1)
        processed_value = integrated_value ** 2

        # Analysis
        if state == 1:
            counter += 1
            if processed_value > max_peak[1]:
                max_peak = (i, processed_value, get_shift(current_window))

            if counter > rr_min + qrs_interval:
                counter = i - max_peak[0]
                state = 2
                r_peaks.append(max_peak[1])
                r_peaks_pos.append(max_peak[0] - max_peak[2])

        elif state == 2:
            counter += 1
            if counter > rr_min:
                th = np.mean(r_peaks)
                state = 3

        elif state == 3:
            if processed_value > th:
                counter = 0
                state = 1
                max_peak = (-1, -1)
            else:
                th = th * math.exp(-p_threshold / 360)

        current_window.pop(0)
        derivated_window.pop(0)
        i += 1

    return r_peaks_pos


def ff_improved_project(ecg):
    # Running median elements
    N = 8
    Nd = 7

    # Preprocessing variables
    padding = max(N, Nd, 10)
    i = 0
    current_window = []
    derivated_window = []

    # Threshold stage constants
    qrs_interval = 21 # 60ms
    rr_min = 72 # 200ms
    p_threshold = 0.7 * 360 / 128 + 4.7

    # Threshold stage variables
    counter = 0
    state = 1
    max_peak = (-1, -1, -1) # (index, value, shift)
    r_peaks = []
    r_peaks_pos = []
    th = 0

    for value in ecg:

        # collecting initial data
        if i < padding:
            current_window.append(value)
            derivated_window.append(value)
            i += 1
            continue

        # Preprocessing
        current_window.append(value)
        derivated_window.append(value - current_window[-Nd])

        integrated_value = np.sum(derivated_window[-1:-1-N:-1]) * 1 / (N - 1)
        processed_value = integrated_value ** 2

        # Analysis
        if state == 1:
            counter += 1
            if processed_value > max_peak[1]:
                max_peak = (i, processed_value, get_shift(current_window))

            if counter > rr_min + qrs_interval:
                if max_peak[1] - 0.056 > processed_value:
                    counter = i - max_peak[0]
                    state = 2
                    r_peaks.append(max_peak[1])
                    r_peaks_pos.append(max_peak[0] - max_peak[2])

        elif state == 2:
            counter += 1
            if counter > rr_min:
                th = np.mean(r_peaks)
                state = 3

        elif state == 3:
            if processed_value > th:
                counter = 0
                state = 1
                max_peak = (-1, -1)
            else:
                th = th * math.exp(-p_threshold / 360)

        current_window.pop(0)
        derivated_window.pop(0)
        i += 1

    return r_peaks_pos


def get_shift(current_window):
    max_elem_idx = max(range(len(current_window)), key=current_window.__getitem__)
    return len(current_window) - max_elem_idx - 1


def ff_my(x):
    Nd = 7
    N = 8
    rr_min = 72 # 200ms, (360 * 0.200)
    qrs_int = 21    # 60ms, (360 * 0.060) TODO: check that value, it is 21.6. Use 21 or 22?
    pth = 6.6
    fs = 360
    exp = -1 * (pth / fs)
    state1_length = rr_min + qrs_int
    y0 = []
    y1 = []
    y = []
    state = 1
    counter = 0
    max_peak_y = 0.0
    max_peak_x = 0
    found_peaks = []
    found_peaks_pos = []
    threshold_amplitude = 0.0
    th = None
    for n in range(0, len(x)):
        n_sub_Nd = max(n - Nd, 0)
        y0.append(x[n] - x[n_sub_Nd])

        sum_y0 = 0.0
        for k in range(0, N):
            n_sub_k = max(n - k, 0)
            sum_y0 += y0[n_sub_k]
        y1.append(sum_y0 / (N - 1))

        y.append(y1[n] * y1[n])

        if 1 == state:
            if y[n] >= max_peak_y:
                max_peak_y = y[n]
                max_peak_x = n
            # at the end of state1:
            counter += 1
            if counter == state1_length:
                # add peak
                found_peaks.append(max_peak_y)
                found_peaks_pos.append(max_peak_x)
                # update amplitude
                threshold_amplitude = sum(found_peaks) / len(found_peaks)
                # change state
                counter = 0
                state = 2
                max_peak_y = 0.0
        elif 2 == state:
            if n - max_peak_x >= rr_min:
                state = 3
        elif 3 == state:
            if th is None:
                th = threshold_amplitude
            else:
                th = th * math.exp(exp)
            if y[n] > th:
                th = None
                state = 1
    # return found_peaks_pos
    return list(map(lambda a: (a, x[a]), found_peaks_pos))