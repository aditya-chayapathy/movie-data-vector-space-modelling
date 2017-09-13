import logging
import math
import time

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
        result = {}
        actor_data = self.get_combined_data_for_actor(actor_id)
        tags = actor_data['tag'].unique()
        for tag in tags:
            sum = 0
            movies = actor_data['movieid'].unique()
            for movie in movies:
                a = self.get_actor_rank_value(actor_id, movie)
                tags_movie_data = actor_data[actor_data['movieid'] == movie]
                tags_of_movie = tags_movie_data['tag'].unique()
                for tag_of_movie in tags_of_movie:
                    m = self.get_model_value(actor_id, movie, tag_of_movie, model)
                    timestamps_tags_movie_data = tags_movie_data['timestamp'].unique
                    for timestamp in timestamps_tags_movie_data:
                        t = self.get_timestamp_value(timestamp)
                        sum = sum + t + m + a
            result[tag] = sum

        return result

    def get_model_value(self, actor_id, movie_id, tag_of_movie, model):
        if model == "tf":
            return self.get_tf_value(actor_id, movie_id, tag_of_movie)
        else:
            return self.get_tfidf_value(actor_id, tag_of_movie)

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

        number_of_divisions = 10
        interval = (maximum - mininum) / number_of_divisions
        value = 0
        upper_bound = mininum - 1
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
        upper_bound = mininum - 1
        while True:
            if actor_rank <= upper_bound:
                break
            upper_bound += interval
            value += 10

        return value


if __name__ == "__main__":
    obj = ActorTag()
    # print obj.get_tf_value(579260, 5857, "true story")
    # print obj.get_idf_value(579260, "violent")
    # print obj.get_timestamp_value("2007-08-27 18:16:41")
    print obj.get_weighted_tags_for_actor_and_model(5857, "tfidf")
    # print obj.get_weighted_tags_for_actor_and_model(579260, "tf")
