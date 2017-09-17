import argparse

parser = argparse.ArgumentParser(
    description='print_genre_vector.py Thriller tfidf',
)
parser.add_argument('genre', action="store", type=str)
parser.add_argument('model', action="store", type=str, choices=set(('tf', 'tfidf')))
input = vars(parser.parse_args())
genre = input['genre']
model = input['model']
print genre, model
