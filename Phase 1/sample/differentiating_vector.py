import logging

import generic_differentiating_vector
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DifferentiatingGenreTag(generic_differentiating_vector.GenericDifferentiateTag):
    def __init__(self, genre1, genre2):
        super(DifferentiatingGenreTag, self).__init__(genre1, genre2)
        self.combined_data = self.get_combined_data()
        self.genres_data = self.get_combined_data_for_genres()
        self.genre1_data = self.get_combined_data_for_object(genre1)
        self.genre2_data = self.get_combined_data_for_object(genre2)
        self.time_utils = utils.TimestampUtils(self.combined_data)
        self.model_utils_genre1 = utils.ModelUtils(self.genre1_data)
        self.model_utils_genre2 = utils.ModelUtils(self.genre2_data)
        self.model_utils_genres = utils.ModelUtils(self.genres_data)

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

    def get_combined_data_for_genres(self):
        result = self.combined_data[
            self.combined_data['genres'].str.contains(self.object_id1) | self.combined_data['genres'].str.contains(
                self.object_id2)]
        return result

    def get_combined_data_for_object(self, genre):
        result = self.combined_data[self.combined_data['genres'].str.contains(genre)]
        return result

    def get_weighted_tags_for_model(self, model):
        row_weights = []
        for index, row in self.genres_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            row_weight = self.get_model_value(movie_id, tag, model)
            row_weights.append(row_weight)

        self.genres_data.is_copy = False
        self.genres_data['row_weight'] = row_weights
        tag_group = self.genres_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df['row_weight'])

        return result

    def get_model_value(self, movie_id, tag_of_movie, model):
        if model == "tfidfdiff":
            return self.get_tfidfdiff_value(movie_id, tag_of_movie) * 100
        elif model == "pdiff1":
            return self.get_pdiff1_value(movie_id, tag_of_movie) * 100
        elif model == "pdiff2":
            return self.get_pdiff2_value(movie_id, tag_of_movie) * 100
        else:
            exit(1)

    def get_pdiff1_value(self, movie_id, tag_of_movie):
        return 1

    def get_pdiff2_value(self, movie_id, tag_of_movie):
        return 2

    def get_tfidfdiff_value(self, movie_id, tag_of_movie):
        return (self.model_utils_genre1.get_tf_value(movie_id, tag_of_movie) - self.model_utils_genre2.get_tf_value(
            movie_id, tag_of_movie)) * self.model_utils_genres.get_idf_value(tag_of_movie)

        # def get_tf_value(self, movie_id, tag_of_movie):
        #     genre_data = self.genre1_data
        #     if genre == self.object_id2:
        #         genre_data = self.genre2_data
        #
        #     doc_data = genre_data[genre_data['movieid'] == movie_id]
        #     tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        #     total_tags_count = doc_data.shape[0]
        #     tag_count = tag_data.shape[0]
        #     if total_tags_count == 0:
        #         return 0
        #     else:
        #         return float(tag_count) / float(total_tags_count)
        #
        # def get_idf_value(self, tag_of_movie):
        #     movies = self.genres_data['movieid'].unique()
        #     doc_count = len(movies)
        #
        #     tag_count = 0
        #     for movie in movies:
        #         movie_data = self.genres_data[self.genres_data['movieid'] == movie]
        #         unique_tags = movie_data['tag'].unique()
        #         for tag in unique_tags:
        #             if tag == tag_of_movie:
        #                 tag_count += 1
        #                 break
        #
        #     return math.log(float(doc_count) / float(tag_count))


if __name__ == "__main__":
    obj = DifferentiatingGenreTag("Thriller", "Children")
    print "TF-IDF-DIFF values for genres 'Thriller' and 'Children':\n"
    result = obj.get_weighted_tags_for_model("tfidfdiff")
    for key, value in sorted(result.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        print "%s: %s" % (key, value)
