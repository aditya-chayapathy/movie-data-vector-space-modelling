import argparse

parser = argparse.ArgumentParser(
    description='print_user_vector.py 4323',
)
parser.add_argument('user_id', action="store", type=int)
input = vars(parser.parse_args())
user_id = input['user_id']
print user_id
