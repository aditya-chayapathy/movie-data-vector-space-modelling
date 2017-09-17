import argparse

import differentiating_vector
import utils

#  Command line interface for task 4
parser = argparse.ArgumentParser(
    description='differentiate_genre.py Thriller Children tfidf',
)
parser.add_argument('genre1', action="store", type=str)
parser.add_argument('genre2', action="store", type=str)
parser.add_argument('model', action="store", type=str, choices=set(('tfidfdiff', 'pdiff1', 'pdiff2')))
input = vars(parser.parse_args())
genre1 = input['genre1']
genre2 = input['genre2']
model = input['model']
print "\n\nTag weights for (%s,%s) in descending order\n\n" % (genre1, genre2)
obj = differentiating_vector.DifferentiatingGenreTag(genre1, genre2)
utils.sort_and_print_dictionary(obj.get_weighted_tags_for_model(model))
