import argparse

parser = argparse.ArgumentParser(
    description='print_user_vector.py <user_id>',
)
parser.add_argument('user_id', action="store", type=int)
input = vars(parser.parse_args())
user_id = input['user_id']
print user_id
