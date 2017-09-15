import argparse

parser = argparse.ArgumentParser(
    description='differentiate_genre.py <genre1> <genre2>',
)
parser.add_argument('genre1', action="store", type=str)
parser.add_argument('genre2', action="store", type=str)
input = vars(parser.parse_args())
genre1 = input['genre1']
genre2 = input['genre2']
print genre1, genre2
