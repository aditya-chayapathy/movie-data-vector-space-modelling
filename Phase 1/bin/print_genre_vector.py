import argparse

parser = argparse.ArgumentParser(
    description='print_genre_vector.py Thriller',
)
parser.add_argument('genre', action="store", type=str)
input = vars(parser.parse_args())
genre = input['genre']
print genre
