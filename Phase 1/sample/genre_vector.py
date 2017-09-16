import logging
import math

import config_parser
import extractor
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config_parser.ParseConfig()


class GenreTag(object):
    def __init__(self, genre):
        self.genre = genre
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.data_extractor = extractor.DataExtractor(self.data_set_location)
        self.combined_data = self.get_combined_data()
        self.time_utils = utils.TimestampUtils(self.combined_data)
        self.genre_data = self.get_combined_data_for_genre()

    def get_combined_data(self):
        mltags = self.data_extractor.get_mltags_data()
        genome_tags = self.data_extractor.get_genome_tags_data()
        mlmovies = self.data_extractor.get_mlmovies_data()

        temp = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del temp['tagId']

        result = temp.merge(mlmovies, on="movieid", how="left")
        del result['userid']
        del result['moviename']
        del result['tagid']

        return result

    def get_combined_data_for_genre(self):
        result = self.combined_data[self.combined_data['genres'].str.contains(self.genre)]
        return result

    def get_weighted_tags_for_model(self, model):
        row_weights = []
        for index, row in self.genre_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            timestamp = row['timestamp']
            row_weight = self.time_utils.get_timestamp_value(timestamp) + self.get_model_value(movie_id, tag, model)
            row_weights.append(row_weight)

        self.genre_data['row_weight'] = row_weights
        tag_group = self.genre_data.groupby(['tag'])
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
        doc_data = self.genre_data[self.genre_data['movieid'] == movie_id]
        tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        total_tags_count = doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, tag_of_movie):
        movies = self.genre_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = self.genre_data[self.genre_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))

    def get_tfidf_value(self, movie_id, tag_of_movie):
        return self.get_tf_value(movie_id, tag_of_movie) * self.get_idf_value(tag_of_movie)


if __name__ == "__main__":
    obj = GenreTag("Thriller")
    print "TF-IDF values for genre : Thriller\n"
    result = obj.get_weighted_tags_for_model("tfidf")
    for key, value in sorted(result.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)
