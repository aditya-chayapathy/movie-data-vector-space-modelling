import argparse

import actor_vector
import utils

parser = argparse.ArgumentParser(
    description='print_actor_vector.py 523452 tfidf',
)
parser.add_argument('actor_id', action="store", type=int)
parser.add_argument('model', action="store", type=str, choices=set(('tf', 'tfidf')))
input = vars(parser.parse_args())
actor_id = input['actor_id']
model = input['model']
print "\n\nTag weights for %s in descending order\n\n" % (actor_id)
obj = actor_vector.ActorTag(actor_id)
utils.sort_and_print_dictionary(obj.get_weighted_tags_for_model(model))
