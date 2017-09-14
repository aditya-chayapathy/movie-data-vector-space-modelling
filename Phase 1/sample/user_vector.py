import logging
import math
import time

import pandas as pd

import config
import extractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config.ParseConfig()


class UserTag(object):
    def __init__(self):
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.extract_data = extractor.ExtractData(self.data_set_location)

    def get_mltags_data(self):
        mltags = self.extract_data.data_extractor("mltags.csv")
        return mltags

    def get_genome_tags_data(self):
        genome_tags = self.extract_data.data_extractor("genome-tags.csv")
        return genome_tags

    def get_combined_data(self):
        mltags = self.get_mltags_data()
        genome_tags = self.get_genome_tags_data()

        result = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del result['tagId']

        return result

    def get_combined_data_for_user(self, user_id):
        user_data = self.get_combined_data()
        result = user_data[user_data['userid'] == user_id]
        return result

    def get_weighted_tags_for_user_and_model(self, user_id, model):
        user_data = self.get_combined_data_for_user(user_id)
        for index, row in user_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            timestamp = row['timestamp']
            row_weight = self.get_timestamp_value(timestamp) + self.get_model_value(user_id, movie_id, tag, model)
            user_data['row_weight'] = pd.Series(row_weight, index=user_data.index)

        tag_group = user_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df['row_weight'])

        return result

    def get_model_value(self, user_id, movie_id, tag_of_movie, model):
        if model == "tf":
            return self.get_tf_value(user_id, movie_id, tag_of_movie) * 100
        elif model == "tfidf":
            return self.get_tfidf_value(user_id, movie_id, tag_of_movie) * 100
        else:
            exit(1)

    def get_tf_value(self, user_id, movie_id, tag_of_movie):
        user_data = self.get_combined_data_for_user(user_id)
        doc_data = user_data[user_data['movieid'] == movie_id]
        tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        total_tags_count = doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, user_data, tag_of_movie):
        user_data = self.get_combined_data_for_user(user_data)
        movies = user_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = user_data[user_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))

    def get_tfidf_value(self, genre, movie_id, tag_of_movie):
        return self.get_tf_value(genre, movie_id, tag_of_movie) * self.get_idf_value(genre, tag_of_movie)

    def get_epoc_timestamp_for_date(self, timestamp):
        return int(time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")))

    def get_timestamp_value(self, timestamp):
        combined_data = self.get_combined_data()
        timestamps = combined_data['timestamp'].unique()
        all_ts = []
        for ts in timestamps:
            all_ts.append(self.get_epoc_timestamp_for_date(ts))

        input_ts = self.get_epoc_timestamp_for_date(timestamp)

        mininum = min(all_ts)
        maximum = max(all_ts)

        number_of_divisions = 100
        interval = (maximum - mininum) / number_of_divisions
        value = 0
        upper_bound = mininum
        while True:
            if input_ts <= upper_bound:
                break
            upper_bound += interval
            value += 1

        return value


if __name__ == "__main__":
    obj = UserTag()
    print "TF-IDF values for user : 146\n"
    result = obj.get_weighted_tags_for_user_and_model(146, "tfidf")
    for key, value in sorted(result.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)
