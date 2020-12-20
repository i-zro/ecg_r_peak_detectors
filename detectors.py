import numpy as np
import math
import statistics
import own_detectors
import scipy.signal as signal

HZ = 360

ALG2_LEVEL_WIDTH = 0.09
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


def alg2_normalize_signal(signal):
    signal_min = 5.0
    signal_max = -5.0
    new_min = 5.0
    new_max = -5.0
    for sample in signal:
        if sample > signal_max:
            signal_max = sample
        if sample < signal_min:
            signal_min = sample
    signal_diff = signal_max - signal_min
    factor = 10.0 / signal_diff
    offset = signal_max * factor - 5.0
    print('signal_max: ' + str(signal_max) + ', signal_min: ' + str(signal_min) +
          ', signal_diff: ' + str(signal_diff) + ', factor: ' + str(factor))
    normalized_signal = []
    for sample in signal:
        new_sample = sample * factor - offset
        if new_sample > new_max:
            new_max = new_sample
        if new_sample < new_min:
            new_min = new_sample
        normalized_signal.append(new_sample)
    print('new_max: ' + str(new_max) + ', new_min: ' + str(new_min))
    return normalized_signal


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
        if x[i] < lower_thres - 0.1 * ALG2_LEVEL_WIDTH:
            events.append(('FALL', i))
            lower_thres -= ALG2_LEVEL_WIDTH
            upper_thres -= ALG2_LEVEL_WIDTH
        elif x[i] > upper_thres + 0.1 * ALG2_LEVEL_WIDTH:
            events.append(('RISE', i))
            lower_thres += ALG2_LEVEL_WIDTH
            upper_thres += ALG2_LEVEL_WIDTH
    # print(events)
    return events


def alg2_chinese(x):
    # x = alg2_normalize_signal(x)
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
    # print('alg3 generating events: first sample value: ' + str(x[0]) + '\tlower_thres: ' + str(lower_thres) + '\tupper_thres:' + str(upper_thres))
    # print('alg3 generating events: first 0.100s (36 samples) values: ' + str(x[:36]))
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
    start_idx = max(p - math.ceil(ALG3_W / 2.0), 0)
    end_idx = min(p + math.ceil(ALG3_W / 2.0) - ALG3_K + ALG3_W % 2, len(events) - 1)
    return int(events[end_idx][1] - events[start_idx][1])


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

# TODO:
def alg3_iranian_test(x):
    SP = None
    NP = None
    events = alg3_generate_events(x)
    # print(events[-100:])
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
                TH1 = (SP + ALG3_COEFF2 * (NP - SP)) + 1
            # print('Alg3: sample: ' + str(events[i][1]) + '\tdur: ' + str(dur) + '\tTH1: ' + str(TH1))
            if dur <= TH1:
                if len(peaks) > 0:
                    if TH2 < abs(events[i][1] - peaks[-1]):
                        SP = SP - ALG3_COEFF1 * (SP - dur)
                        pob = min(pob - ALG3_COEFF3 * (pob - abs(events[i][1] - peaks[-1])), 360)
                        TH2 = 0.5 * pob
                        peaks.append(events[i][1])
                        TH1 = SP + ALG3_COEFF2 * (NP - SP)
                else:
                    SP = SP - ALG3_COEFF1 * (SP - dur)
                    peaks.append(events[i][1])
                    TH1 = 1 + (SP + ALG3_COEFF2 * (NP - SP))
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


def alg5_pan_tompkins(x):
    return pan_tompkins_detector(x)

fs = 360
def pan_tompkins_detector(unfiltered_ecg, MWA_name='cumulative'):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.
    """

    f1 = 5 / fs
    f2 = 15 / fs

    b, a = signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')

    filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

    diff = np.diff(filtered_ecg)

    squared = diff * diff

    N = int(0.12 * fs)
    mwa = MWA_from_name(MWA_name)(squared, N)
    mwa[:int(0.2 * fs)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks


def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else:
        raise RuntimeError('invalid moving average function!')


# Fast implementation of moving window average with numpy's cumsum function
def MWA_cumulative(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


# Original Function
def MWA_original(input_array, window_size):
    mwa = np.zeros(len(input_array))
    mwa[0] = input_array[0]

    for i in range(2, len(input_array) + 1):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i - window_size:i]

        mwa[i - 1] = np.mean(section)

    return mwa


# Fast moving window average implemented with 1D convolution
def MWA_convolve(input_array, window_size):
    ret = np.pad(input_array, (window_size - 1, 0), 'constant', constant_values=(0, 0))
    ret = np.convolve(ret, np.ones(window_size), 'valid')

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks


def alg_own(x):
    return own_detectors.current(x)
