import argparse

parser = argparse.ArgumentParser(
    description='print_actor_vector.py 523452',
)
parser.add_argument('actor_id', action="store", type=int)
input = vars(parser.parse_args())
actor_id = input['actor_id']
print actor_id
