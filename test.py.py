import ast
import argparse


def str_eval(string):
    print('string:')
    print(string)
    return ast.literal_eval(string)

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str_eval,default=None,help="initial input prompt")
args = parser.parse_args()

print(str('[1,2,3]')[0])