import argparse

from database import Database
from dataset import Dataset

def parse_args():
    """Function to parse the arguments of the program."""
    parser = argparse.ArgumentParser()


    parser.add_argument('--db', type=str, required=True, help="Path to the database. REQUIRED")
    # parser.add_argument('--image', type=str, nargs='+', required=False, help="Path to one or more images. Mutually exclusive with '--dataset'")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset. Mutually exclusive with '--image'")
    
    args = parser.parse_args()
    check_args(args)
    return args

def check_args(args: argparse.Namespace):
    """Function to check all arguments are correct."""

    pass

def main():
    """Main function of the program."""
    args = parse_args() 

    # First. Load the database where we will extract our information from. Precompute all histograms.
    db = Database(args.db)

    # Second. Load the image/dataset we want to process.
    data = Dataset(args.dataset)
    
    
    # Three. Extract descriptor (histogram) of the image. It is done as soon as the images are loaded


    # Four. Compare histogram distances.


    # Five. Choose top K images


    pass

if __name__ == "__main__":
    main()

