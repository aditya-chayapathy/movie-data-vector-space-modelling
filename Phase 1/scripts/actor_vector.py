import logging

import generic_vector
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ActorTag(generic_vector.GenericTag):  # Represents a class to establish relationship between tags and actors
    def __init__(self, actor_id):
        super(ActorTag, self).__init__(actor_id)
        self.combined_data = self.get_combined_data()
        self.actor_data = self.get_combined_data_for_object()
        self.time_utils = utils.TimestampUtils(self.combined_data)
        self.model_utils = utils.ModelUtils(self.combined_data)

    def get_combined_data(self):  # complete data set
        mltags = self.data_extractor.get_mltags_data()
        genome_tags = self.data_extractor.get_genome_tags_data()
        movie_actor = self.data_extractor.get_movie_actor_data()

        temp = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del temp['tagId']

        result = temp.merge(movie_actor, on="movieid", how="left")

        return result

    def get_combined_data_for_object(self):  # actor specific data
        result = self.combined_data[self.combined_data["actorid"] == self.object_id]
        del result['actorid']
        return result

    def get_weighted_tags_for_model(self, model):
        row_weights = []
        for index, row in self.actor_data.iterrows():  # for each row in the actor data set
            movie_id = row['movieid']
            tag = row['tag']
            timestamp = row['timestamp']
            row_weight = utils.MovieUtils(self.combined_data, movie_id,
                                          self.object_id).get_actor_rank_value() + self.time_utils.get_timestamp_value(
                timestamp) + self.get_model_value(movie_id, tag,
                                                  model)  # row weight = model weight + timestamp weight + actor rank weight
            row_weights.append(row_weight)

        self.actor_data.is_copy = False
        self.actor_data['row_weight'] = row_weights
        tag_group = self.actor_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df[
                                  'row_weight'])  # calculate final tag value by aggregating (summing) individual tags values encountered in the data set

        return result

    def get_model_value(self, movie_id, tag_of_movie, model):  # obtain the value for the model passed as input
        if model == "tf":
            return self.model_utils.get_tf_value(movie_id, tag_of_movie) * 100
        elif model == "tfidf":
            return self.model_utils.get_tfidf_value(movie_id, tag_of_movie) * 100
        else:
            exit(1)


if __name__ == "__main__":
    obj = ActorTag(579260)
    print "TF-IDF values for actor DiCaprio (actor_id:579260)\n"
    result = obj.get_weighted_tags_for_model("tfidf")
    utils.sort_and_print_dictionary(result)
