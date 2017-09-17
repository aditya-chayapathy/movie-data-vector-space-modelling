import argparse

import genre_vector
import utils

parser = argparse.ArgumentParser(
    description='print_genre_vector.py Thriller tfidf',
)
parser.add_argument('genre', action="store", type=str)
parser.add_argument('model', action="store", type=str, choices=set(('tf', 'tfidf')))
input = vars(parser.parse_args())
genre = input['genre']
model = input['model']
print "\n\nTag weights for %s in descending order\n\n" % (genre)
obj = genre_vector.GenreTag(genre)
utils.sort_and_print_dictionary(obj.get_weighted_tags_for_model(model))
