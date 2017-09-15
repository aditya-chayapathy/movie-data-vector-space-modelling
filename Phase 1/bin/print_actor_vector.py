import argparse

parser = argparse.ArgumentParser(
    description='print_actor_vector.py 523452 tfidf',
)
parser.add_argument('actor_id', action="store", type=int)
parser.add_argument('model', action="store", type=str, choices=set(('tf', 'tfidf')))
input = vars(parser.parse_args())
actor_id = input['actor_id']
model = input['model']
print actor_id, model
