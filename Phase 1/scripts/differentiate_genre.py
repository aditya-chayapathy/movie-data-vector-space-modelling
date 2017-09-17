import argparse

import

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
obj = differentiating_vector.DifferentiatingGenreTag('asd', "asd")
print genre1, genre2, model
