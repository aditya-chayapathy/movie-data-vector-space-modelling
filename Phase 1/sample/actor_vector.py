import logging
import math

import config_parser
import extractor
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config_parser.ParseConfig()


class ActorTag(object):
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.data_extractor = extractor.DataExtractor(self.data_set_location)
        self.combined_data = self.get_combined_data()
        self.time_utils = utils.TimestampUtils(self.combined_data)
        self.actor_data = self.get_combined_data_for_actor()

    def get_combined_data(self):
        mltags = self.data_extractor.get_mltags_data()
        genome_tags = self.data_extractor.get_genome_tags_data()
        movie_actor = self.data_extractor.get_movie_actor_data()

        temp = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del temp['tagId']

        result = temp.merge(movie_actor, on="movieid", how="left")

        return result

    def get_combined_data_for_actor(self):
        result = self.combined_data[self.combined_data["actorid"] == self.actor_id]
        del result['actorid']
        return result

    def get_weighted_tags_for_model(self, model):
        row_weights = []
        for index, row in self.actor_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            timestamp = row['timestamp']
            row_weight = self.get_actor_rank_value(movie_id) + self.time_utils.get_timestamp_value(
                timestamp) + self.get_model_value(movie_id, tag, model)
            row_weights.append(row_weight)

        self.actor_data['row_weight'] = row_weights
        tag_group = self.actor_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df['row_weight'])

        return result

    def get_model_value(self, movie_id, tag_of_movie, model):
        if model == "tf":
            return self.get_tf_value(movie_id, tag_of_movie) * 100
        elif model == "tfidf":
            return self.get_tfidf_value(movie_id, tag_of_movie) * 100
        else:
            exit(1)

    def get_tf_value(self, movie_id, tag_of_movie):
        doc_data = self.actor_data[self.actor_data['movieid'] == movie_id]
        tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        total_tags_count = doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, tag_of_movie):
        movies = self.actor_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = self.actor_data[self.actor_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))

    def get_tfidf_value(self, movie_id, tag_of_movie):
        return self.get_tf_value(movie_id, tag_of_movie) * self.get_idf_value(tag_of_movie)

    def get_actor_rank_value(self, movie_id):
        movie_data = self.combined_data[self.combined_data['movieid'] == movie_id]
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


if __name__ == "__main__":
    obj = ActorTag(579260)
    print "TF-IDF values for actor DiCaprio (actor_id:579260)\n"
    result = obj.get_weighted_tags_for_model("tfidf")
    utils.sort_and_print_dictionary(result)
