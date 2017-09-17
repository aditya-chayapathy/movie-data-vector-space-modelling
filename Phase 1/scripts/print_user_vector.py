import argparse

import user_vector
import utils

parser = argparse.ArgumentParser(
    description='print_user_vector.py 4323 tfidf',
)
parser.add_argument('user_id', action="store", type=int)
parser.add_argument('model', action="store", type=str, choices=set(('tf', 'tfidf')))
input = vars(parser.parse_args())
user_id = input['user_id']
model = input['model']
print "\n\nTag weights for %s in descending order\n\n" % (user_id)
obj = user_vector.UserTag(user_id)
utils.sort_and_print_dictionary(obj.get_weighted_tags_for_model(model))
