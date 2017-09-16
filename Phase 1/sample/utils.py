import math
import time


def get_epoc_timestamp_for_date(timestamp):
    return int(time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")))


def sort_and_print_dictionary(dict1):
    for key, value in sorted(dict1.iteritems(), key=lambda (k, v): (v, k), reverse=True):
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


class ModelUtils(object):
    def __init__(self, object_data):
        self.object_data = object_data

    def get_tf_value(self, movie_id, tag_of_movie):
        object_doc_data = self.object_data[self.object_data['movieid'] == movie_id]
        tag_data = object_doc_data[object_doc_data['tag'] == tag_of_movie]
        total_tags_count = object_doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, tag_of_movie):
        movies = self.object_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = self.object_data[self.object_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))

    def get_tfidf_value(self, movie_id, tag_of_movie):
        return self.get_tf_value(movie_id, tag_of_movie) * self.get_idf_value(tag_of_movie)
