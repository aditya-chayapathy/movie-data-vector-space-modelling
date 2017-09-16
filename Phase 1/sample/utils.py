import time


def get_epoc_timestamp_for_date(timestamp):
    return int(time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")))


def sort_and_print_dictionary(dict):
    for key, value in sorted(dict.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)


class TimestampUtils(object):
    def __init__(self, combined_data):
        self.combined_data = combined_data
        timestamps = combined_data['timestamp'].unique()
        all_ts = []
        for ts in timestamps:
            all_ts.append(get_epoc_timestamp_for_date(ts))
        self.min_ts = min(all_ts)
        self.max_ts = max(all_ts)

    def get_timestamp_value(self, timestamp):
        input_ts = get_epoc_timestamp_for_date(timestamp)
        number_of_divisions = 100
        interval = (self.max_ts - self.min_ts) / number_of_divisions
        value = 0.0
        upper_bound = self.min_ts
        while True:
            if input_ts <= upper_bound:
                break
            upper_bound += interval
            value += 0.01
        return value * 10
