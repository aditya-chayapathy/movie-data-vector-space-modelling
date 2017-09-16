import logging
import math

import config_parser
import extractor
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config_parser.ParseConfig()


class UserTag(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.data_extractor = extractor.DataExtractor(self.data_set_location)
        self.combined_data = self.get_combined_data()
        self.time_utils = utils.TimestampUtils(self.combined_data)
        self.user_data = self.get_combined_data_for_user(self.user_id)

    def get_combined_data(self):
        mltags = self.data_extractor.get_mltags_data()
        genome_tags = self.data_extractor.get_genome_tags_data()

        result = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del result['tagId']

        return result

    def get_combined_data_for_user(self, user_id):
        result = self.combined_data[self.combined_data['userid'] == user_id]
        return result

    def get_weighted_tags_for_model(self, model):
        row_weights = []
        for index, row in self.user_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            timestamp = row['timestamp']
            row_weight = self.time_utils.get_timestamp_value(timestamp) * self.get_model_value(movie_id, tag,
                                                                                               model) * 100
            row_weights.append(row_weight)

        self.user_data['row_weight'] = row_weights
        tag_group = self.user_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df['row_weight'])

        return result

    def get_model_value(self, movie_id, tag_of_movie, model):
        if model == "tf":
            return self.get_tf_value(movie_id, tag_of_movie)
        elif model == "tfidf":
            return self.get_tfidf_value(movie_id, tag_of_movie)
        else:
            exit(1)

    def get_tf_value(self, movie_id, tag_of_movie):
        doc_data = self.user_data[self.user_data['movieid'] == movie_id]
        tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        total_tags_count = doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, tag_of_movie):
        movies = self.user_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = self.user_data[self.user_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))

    def get_tfidf_value(self, movie_id, tag_of_movie):
        return self.get_tf_value(movie_id, tag_of_movie) * self.get_idf_value(tag_of_movie)


if __name__ == "__main__":
    obj = UserTag(146)
    print "TF-IDF values for user : 146\n"
    result = obj.get_weighted_tags_for_model("tf")
    for key, value in sorted(result.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)
