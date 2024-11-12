import argparse
import os
import sys
import train


def parse_arguments():
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Path in the filesystem to a directory where data and metadata CSV files are located.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path in the filesystem to a directory where processed model data will be exported.')

    return parser.parse_args()


def check_parameters(args):
    '''
    Check if parameters are valid
    '''
    if not os.path.isdir(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.output):
        print(f"Error: Output directory '{args.output}' does not exist.")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Check parameters
    check_parameters(args)

    train.run(args)