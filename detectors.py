import numpy as np
import math
import statistics
import own_detectors

HZ = 360

ALG2_LEVEL_WIDTH = 0.07
ALG2_A_THRES = 7

ALG3_LEVEL_WIDTH = 0.078
ALG3_K = 4
ALG3_W = 9
ALG3_RISE_AFTER_FALL = 0
ALG3_RISE_AFTER_RISE = 1
ALG3_FALL_AFTER_FALL = 2
ALG3_FALL_AFTER_RISE = 3
ALG3_COEFF1 = 0.25
ALG3_COEFF2 = 0.25
ALG3_COEFF3 = 0.125


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


def alg1_spanish(x):
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
    return found_peaks_pos


def alg2_get_thresholds(val):
    lower_thres = 0.0
    upper_thres = ALG2_LEVEL_WIDTH
    while val < lower_thres:
        lower_thres -= ALG2_LEVEL_WIDTH
        upper_thres -= ALG2_LEVEL_WIDTH
    while val > upper_thres:
        lower_thres += ALG2_LEVEL_WIDTH
        upper_thres += ALG2_LEVEL_WIDTH
    return lower_thres, upper_thres


def alg2_generate_events(x):
    lower_thres, upper_thres = alg2_get_thresholds(x[0])
    # print('ecg[0]: ' + str(x[0]) + ', lower_thres: ' + str(lower_thres) + ', upper_thres: ' + str(upper_thres))
    events = []
    x_len = len(x)
    for i in range(0, x_len):
        if x[i] < lower_thres:
            events.append(('FALL', i))
            lower_thres -= ALG2_LEVEL_WIDTH
            upper_thres -= ALG2_LEVEL_WIDTH
        elif x[i] > upper_thres:
            events.append(('RISE', i))
            lower_thres += ALG2_LEVEL_WIDTH
            upper_thres += ALG2_LEVEL_WIDTH
    # print(events)
    return events


def alg2_chinese(x):
    events = alg2_generate_events(x)
    step = 1
    counter_rise = 0
    counter_fall = 0
    r_quess = 0
    found_peaks = []
    prev_event = events[0][0]
    for event in events[1:]:
        if 1 == step:
            if event[0] == 'RISE':
                counter_rise += 1
            else:
                counter_rise = 0
            if counter_rise > ALG2_A_THRES: # TODO: > or >=
                counter_rise = 0
                step = 2
        elif 2 == step:
            if 0 == counter_fall:
                if event[0] == 'FALL':
                    r_quess = int((prev_event[1] + event[1]) / 2)
                    counter_fall += 1
            else:
                if event[0] == 'FALL':
                    counter_fall += 1
                    if counter_fall > ALG2_A_THRES: # TODO: > or >=
                        counter_fall = 0
                        step = 3
                else:
                    counter_fall = 0
                    step = 1
                    counter_rise = 1
        elif 3 == step:
            if event[0] == 'RISE':
                step = 1
                found_peaks.append(r_quess)
        prev_event = event
    return found_peaks


def alg3_get_thresholds(val):
    lower_thres = 0.0
    upper_thres = ALG3_K * ALG3_LEVEL_WIDTH
    while val < lower_thres:
        lower_thres -= ALG3_LEVEL_WIDTH
        upper_thres -= ALG3_LEVEL_WIDTH
    while val > upper_thres:
        lower_thres += ALG3_LEVEL_WIDTH
        upper_thres += ALG3_LEVEL_WIDTH
    return lower_thres, upper_thres


# TODO: refactor
def alg3_generate_events(x):
    lower_thres, upper_thres = alg3_get_thresholds(x[0])
    events = []
    x_len = len(x)
    for i in range(0, x_len):
        if x[i] < lower_thres:
            if 0 == len(events):
                events.append((ALG3_FALL_AFTER_FALL, i))
            else:
                if ALG3_RISE_AFTER_FALL == events[-1][0] or ALG3_RISE_AFTER_RISE == events[-1][0]:
                    events.append((ALG3_FALL_AFTER_RISE, i))
                else:
                    events.append((ALG3_FALL_AFTER_FALL, i))
            lower_thres -= ALG2_LEVEL_WIDTH
            upper_thres -= ALG2_LEVEL_WIDTH
        elif x[i] > upper_thres:
            if 0 == len(events):
                events.append((ALG3_RISE_AFTER_RISE, i))
            else:
                if ALG3_FALL_AFTER_FALL == events[-1][0] or ALG3_FALL_AFTER_RISE == events[-1][0]:
                    events.append((ALG3_RISE_AFTER_FALL, i)) # 0 - RISE after FALL
                else:
                    events.append((ALG3_RISE_AFTER_RISE, i))
            lower_thres += ALG2_LEVEL_WIDTH
            upper_thres += ALG2_LEVEL_WIDTH
    # print(events)
    return events


def is_peak(event_type):
    return ALG3_FALL_AFTER_RISE == event_type or ALG3_RISE_AFTER_FALL == event_type


def get_dur(events, p):
    peak_time = events[p][1]
    start_idx = p - int(ALG3_W / 2)
    end_idx = p + int(ALG3_W / 2) - ALG3_K + ALG3_W % 2
    dur = 0
    for i in range(start_idx, end_idx):
        if 0 <= i <= len(events):
            dur += abs(events[i][1] - peak_time)
    return dur


# TODO:
def alg3_iranian(x):
    SP = None
    NP = None
    events = alg3_generate_events(x)
    events_len = len(events)
    peaks = []
    pob = 300
    TH2 = 0.5 * pob
    for i in range(0, events_len):
        if is_peak(events[i][0]):
            dur = get_dur(events, i)
            if SP is None:
                SP = dur
                NP = dur
                TH1 = SP + ALG3_COEFF2 * (NP - SP)
            if dur < TH1:
                if len(peaks) > 0:
                    if TH2 < abs(events[i][1] - peaks[-1]):
                        pob = min(pob - ALG3_COEFF3 * (pob - abs(events[i][1] - peaks[-1])), 360)
                        TH2 = 0.5 * pob
                        peaks.append(events[i][1])
                else:
                    peaks.append(events[i][1])
                SP = SP - ALG3_COEFF1 * (SP - dur)
            else:
                NP = NP - ALG3_COEFF1 * (NP - dur)
    print('Tape evaluated by alg3 iranian')
    return peaks


def alg4_polish(x):
    ALPHA = 0.46
    GAMMA = 0.97
    samples_num_short_avg = 20
    samples_num_long_avg = 100
    samples_num_window = 72
    max_diff_arg, max_diff_val = alg4_find_first_peak(x)
    found_peaks = [max_diff_arg]
    threshold = ALPHA * max_diff_val
    short_avg_sum = 0.0
    long_avg_sum = 0.0
    search_samples_left = 0
    max_x = 0
    max_abs_y = 0.0
    refractory_window_end = found_peaks[0] + samples_num_window
    is_inside_refractory_window = True
    is_inside_searching_window = False
    x_len = len(x)
    max_new = 0.0
    for i in range(0, x_len):
        short_avg_sum += x[i]
        long_avg_sum += x[i]
        if i >= samples_num_short_avg:
            short_avg_sum -= x[i - samples_num_short_avg]
        if i >= samples_num_long_avg:
            long_avg_sum -= x[i - samples_num_long_avg]
        if i < samples_num_short_avg - 1:
            continue

        abs_diff_short = abs(short_avg_sum / samples_num_short_avg - x[i]) #TODO: czy tu na pewno ten short diff
        if is_inside_refractory_window:
            if i == refractory_window_end:
                is_inside_refractory_window = False
            else:
                continue
        if abs_diff_short >= threshold:
            if not is_inside_searching_window:
                is_inside_searching_window = True
                search_samples_left = samples_num_window
        if is_inside_searching_window:
            if abs(long_avg_sum / samples_num_long_avg - x[i]) > max_abs_y:
                max_x = i
                max_abs_y = abs(long_avg_sum / samples_num_long_avg - x[i])
                max_abs_short = abs_diff_short
            if abs(short_avg_sum / samples_num_short_avg - x[i]) > max_new:
                max_new = abs(short_avg_sum / samples_num_short_avg - x[i])
            search_samples_left -= 1
            if search_samples_left == 0:
                found_peaks.append(max_x)
                threshold = GAMMA * threshold + ALPHA * (1 - GAMMA) * max_new #TODO: z czego tu ten maks
                is_inside_searching_window = False
                is_inside_refractory_window = True
                refractory_window_end = max_x + samples_num_window
                max_x = 0
                max_new = 0.0
                max_abs_y = 0.0
    print('Tape evaluated by alg4')
    return found_peaks


def alg4_find_first_peak(x):
    short_avg_sum = 0.0
    max_diff_val = -1.0
    max_diff_arg = 0
    for i in range(0, HZ):
        short_avg_sum += x[i]
        if i < 20 - 1:
            continue
        abs_diff_short = abs(short_avg_sum / 20 - x[i])
        if abs_diff_short > max_diff_val:
            max_diff_val = abs_diff_short
            max_diff_arg = i
        short_avg_sum -= x[i - 20]
    return max_diff_arg, max_diff_val


def alg_own(x):
    return own_detectors.current(x)
