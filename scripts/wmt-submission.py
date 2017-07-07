#!/usr/bin/env python3
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('system')
parser.add_argument('--institution', default='LIG')
parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--output-dir')

if __name__ == '__main__':
    args = parser.parse_args()

    type_ = 'CONTRASTIVE' if args.contrastive else 'PRIMARY'
    filename = '{}_{}_{}'.format(args.institution, args.system, type_)

    if args.output_dir:
        filename = os.path.join(args.output_dir, filename)

    with open(args.filename) as input_file, open(filename, 'w') as output_file:
        for i, line in enumerate(input_file, 1):
            output_file.write('{}\t{}\t{}'.format(args.system, i, line))
