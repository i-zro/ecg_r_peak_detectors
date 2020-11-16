SAMPLES_PER_SECOND = 360


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


def time_to_sample(time):
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