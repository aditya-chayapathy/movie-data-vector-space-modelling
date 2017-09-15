import logging
import math

import pandas as pd

import config
import extractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = config.ParseConfig()


class DifferentiatingGenreTag(object):
    def __init__(self):
        self.data_set_location = conf.config_section_mapper("filePath").get("data_set_location")
        self.extract_data = extractor.ExtractData(self.data_set_location)

    def get_mlmovies_data(self):
        mlmovies = self.extract_data.data_extractor("mlmovies.csv")
        return mlmovies

    def get_mltags_data(self):
        mltags = self.extract_data.data_extractor("mltags.csv")
        return mltags

    def get_genome_tags_data(self):
        genome_tags = self.extract_data.data_extractor("genome-tags.csv")
        return genome_tags

    def get_combined_data(self):
        mltags = self.get_mltags_data()
        genome_tags = self.get_genome_tags_data()
        mlmovies = self.get_mlmovies_data()

        temp = mltags.merge(genome_tags, left_on="tagid", right_on="tagId", how="left")
        del temp['tagId']

        result = temp.merge(mlmovies, on="movieid", how="left")
        del result['userid']
        del result['moviename']
        del result['tagid']

        return result

    def get_combined_data_for_genres(self, genre1, genre2):
        genre_data = self.get_combined_data()
        result = genre_data[genre_data['genres'].str.contains(genre1) | genre_data['genres'].str.contains(genre2)]
        return result

    def get_combined_data_for_genre(self, genre):
        genre_data = self.get_combined_data()
        result = genre_data[genre_data['genres'].str.contains(genre)]
        return result

    def get_weighted_tags_for_genre_and_model(self, genre1, genre2, model):
        genre_data = self.get_combined_data_for_genres(genre1, genre2)
        for index, row in genre_data.iterrows():
            movie_id = row['movieid']
            tag = row['tag']
            row_weight = self.get_model_value(genre1, genre2, movie_id, tag, model)
            genre_data['row_weight'] = pd.Series(row_weight, index=genre_data.index)

        tag_group = genre_data.groupby(['tag'])
        result = {}
        for tag, df in tag_group:
            result[tag] = sum(df['row_weight'])

        return result

    def get_model_value(self, genre1, genre2, movie_id, tag_of_movie, model):
        if model == "tfidfdiff":
            return self.get_tfidfdiff_value(genre1, genre2, movie_id, tag_of_movie) * 100
        elif model == "pdiff1":
            return self.get_pdiff1_value(genre1, genre2, movie_id, tag_of_movie) * 100
        elif model == "pdiff2":
            return self.get_pdiff2_value(genre1, genre2, movie_id, tag_of_movie) * 100
        else:
            exit(1)

    def get_tfidfdiff_value(self, genre1, genre2, movie_id, tag_of_movie):
        return self.get_tf_value(genre1, movie_id, tag_of_movie) * self.get_idf_value(genre1, genre2, tag_of_movie)

    def get_tf_value(self, genre, movie_id, tag_of_movie):
        genre_data = self.get_combined_data_for_genre(genre)
        doc_data = genre_data[genre_data['movieid'] == movie_id]
        tag_data = doc_data[doc_data['tag'] == tag_of_movie]
        total_tags_count = doc_data.shape[0]
        tag_count = tag_data.shape[0]
        return float(tag_count) / float(total_tags_count)

    def get_idf_value(self, genre1, genre2, tag_of_movie):
        genre_data = self.get_combined_data_for_genres(genre1, genre2)
        movies = genre_data['movieid'].unique()
        doc_count = len(movies)

        tag_count = 0
        for movie in movies:
            movie_data = genre_data[genre_data['movieid'] == movie]
            unique_tags = movie_data['tag'].unique()
            for tag in unique_tags:
                if tag == tag_of_movie:
                    tag_count += 1
                    break

        return math.log(float(doc_count) / float(tag_count))


if __name__ == "__main__":
    obj = DifferentiatingGenreTag()
    print "TF-IDF values for genre : Thriller and genre : Children\n"
    print obj.get_combined_data_for_genres("Thriller", "Children")
    # result = obj.get_weighted_tags_for_genres_and_model("Thriller", "Children", "tfidf")
    # for key, value in sorted(result.iteritems(), key=lambda (k, v): (v, k), reverse=True):
    #     print "%s: %s" % (key, value)
