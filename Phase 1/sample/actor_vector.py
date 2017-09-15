import logging
import math
import time

import pandas as pd

import config
import extractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config.ParseConfig()


class ActorTag(object):
    def __init__(self):
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.extract_data = extractor.ExtractData(self.data_set_location)

    def get_movie_actor_data(self):
        movie_actor = self.extract_data.data_extractor("movie-actor.csv")
        return movie_actor

    def get_mltags_data(self):
        mltags = self.extract_data.data_extractor("mltags.csv")
        return mltags

    def get_genome_tags_data(self):
        genome_tags = self.extract_data.data_extractor("genome-tags.csv")
        return genome_tags

    def get_combined_data(self):
        mltags = self.get_mltags_data()
        genome_tags = self.get_genome_tags_data()
        movie_actor = self.get_movie_actor_data()

        temp = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del temp['tagId']

        result = temp.merge(movie_actor, on="movieid", how="left")

        return result

    def get_combined_data_for_actor(self, actor_id):
        actor_data = self.get_combined_data()
        result = actor_data.loc[actor_data["actorid"] == actor_id]
        del result['actorid']
        return result

    def get_weighted_tags_for_actor_and_model(self, actor_id, model):
        actor_data = self.get_combined_data_for_actor(actor_id)
        for index, row in actor_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            timestamp = row['timestamp']
            row_weight = self.get_actor_rank_value(actor_id, movie_id) + self.get_timestamp_value(
                timestamp) + self.get_model_value(actor_id, movie_id, tag, model)
            actor_data['row_weight'] = pd.Series(row_weight, index=actor_data.index)

        tag_group = actor_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df['row_weight'])

        return result

    def get_model_value(self, actor_id, movie_id, tag_of_movie, model):
        if model == "tf":
            return self.get_tf_value(actor_id, movie_id, tag_of_movie) * 1000
        elif model == "tfidf":
            return self.get_tfidf_value(actor_id, movie_id, tag_of_movie) * 1000
        else:
            exit(1)

    def get_tf_value(self, actor_id, movie_id, tag_of_movie):
        actor_data = self.get_combined_data_for_actor(actor_id)
        doc_data = actor_data[actor_data['movieid'] == movie_id]
        tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        total_tags_count = doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, actor_id, tag_of_movie):
        actor_data = self.get_combined_data_for_actor(actor_id)
        movies = actor_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = actor_data[actor_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))

    def get_tfidf_value(self, actor_id, movie_id, tag_of_movie):
        return self.get_tf_value(actor_id, movie_id, tag_of_movie) * self.get_idf_value(actor_id, tag_of_movie)

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

    def get_actor_rank_value(self, actor_id, movie_id):
        combined_data = self.get_combined_data()
        movie_data = combined_data[combined_data['movieid'] == movie_id]
        ranks = movie_data['actor_movie_rank'].unique()
        mininum = min(ranks)
        maximum = max(ranks)

        number_of_divisions = 10
        interval = (maximum - mininum) / number_of_divisions
        actor_rank = movie_data[movie_data['actorid'] == actor_id]['actor_movie_rank'].unique()[0]

        value = 0
        upper_bound = mininum
        while True:
            if actor_rank <= upper_bound:
                break
            upper_bound += interval
            value += 10

        return 100 - value


if __name__ == "__main__":
    obj = ActorTag()
    print "TF-IDF values for actor DiCaprio (actor_id:579260)\n"
    result = obj.get_weighted_tags_for_actor_and_model(579260, "tfidf")
    for key, value in sorted(result.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)
