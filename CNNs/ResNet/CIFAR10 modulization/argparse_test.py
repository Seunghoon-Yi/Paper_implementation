import argparse
import os

parser = argparse.ArgumentParser(description='argparse tutorial')

# Add arguments :
parser.add_argument('--print-number', type=int, default=5,
                    help='an integer for printing 1 to given number')
parser.add_argument('--square', type=int, default=12,
                    help='an integer for printing squared value of it')

# Arguments to dictionary :
args = parser.parse_args()

for i in range(args.print_number):
    print(f'print number : {i+1}')

print(f"{args.square}, squared value = {args.square**2}")