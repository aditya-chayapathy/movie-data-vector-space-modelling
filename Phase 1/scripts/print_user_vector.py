import argparse

parser = argparse.ArgumentParser(
    description='print_user_vector.py 4323 tfidf',
)
parser.add_argument('user_id', action="store", type=int)
parser.add_argument('model', action="store", type=str, choices=set(('tf', 'tfidf')))
input = vars(parser.parse_args())
user_id = input['user_id']
model = input['model']
print user_id, model
