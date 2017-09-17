import math
import time


def get_epoc_timestamp_for_date(timestamp):  # Convert timestamp from string to epoc
    return int(time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")))


def sort_and_print_dictionary(dict1):  # Print the dictionary in descending order of weights
    for key, value in sorted(dict1.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)


class TimestampUtils(object):  # contains all the timestamp related utilities
    def __init__(self, combined_data):
        self.combined_data = combined_data
        timestamps = combined_data['timestamp'].unique()
        all_ts = []
        for ts in timestamps:
            all_ts.append(get_epoc_timestamp_for_date(ts))
        self.min_ts = min(all_ts)
        self.max_ts = max(all_ts)

    def get_timestamp_value(self,
                            timestamp):  # returns a value between 1 to 10 that represent the weight associated with a timestamp. Newer timestamps are given higher values. The newest timestamp is given a value of 10 while the oldest timestamp us given a value of 0.
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


class ModelUtils(object):  # contains all the utilities related to TF/TF-IDF calculations
    def __init__(self, object_data):
        self.object_data = object_data

    def get_tf_value(self, movie_id, tag_of_movie):  # calculate TF
        object_doc_data = self.object_data[self.object_data['movieid'] == movie_id]
        tag_data = object_doc_data[object_doc_data['tag'] == tag_of_movie]
        total_tags_count = object_doc_data.shape[0]
        tag_count = tag_data.shape[0]
        if total_tags_count == 0:
            return 0
        return float(tag_count) / float(total_tags_count)  # no_of_terms_in_document / total_number_of_terms_in_document

    def get_idf_value(self, tag_of_movie):  # calculate IDF
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

        return math.log(float(doc_count) / float(
            tag_count))  # total_number_of_documents / total_number_document_containing_the_term

    def get_tfidf_value(self, movie_id, tag_of_movie):  # calculate TF-IDF
        return self.get_tf_value(movie_id, tag_of_movie) * self.get_idf_value(tag_of_movie)  # tf * idf


class MovieUtils(object):  # contains all the utilities related to movies
    def __init__(self, combined_data, movie_id, actor_id):
        self.movie_id = movie_id
        self.combined_data = combined_data
        self.actor_id = actor_id

    def get_actor_rank_value(
            self):  # returns a value between 1 to 10 that represent the weight associated with an actor for a particular movie. Lower actor ranks are given higher values. For a movie, the actor with the least rank is given a value 10 while the actor with the highest rank is given a value of 0.
        movie_data = self.combined_data[self.combined_data['movieid'] == self.movie_id]
        ranks = movie_data['actor_movie_rank'].unique()
        min_rank = min(ranks)
        max_rank = max(ranks)

        number_of_divisions = 10
        interval = (max_rank - min_rank) / number_of_divisions
        actor_rank = movie_data[movie_data['actorid'] == self.actor_id]['actor_movie_rank'].unique()[0]

        value = 0.0
        upper_bound = min_rank
        while True:
            if actor_rank <= upper_bound:
                break
            upper_bound += interval
            value += 0.1

        return (1.0 - value) * 10
