# C1 Project
Welcome to the Team1's project of the C1 course from the MCV 25-26!

With this project, given a certain database of images (i.e. art from a museum) you can retrieve information of each element of the database with another image taken by you. This technique is called Content Based Image Retrieval (CBIR) and as for now it just uses histograms of the database and the query image to find the right match and provide the relevant information. (Be aware that the algorithm is not perfect and it can fail and retrieve a different image by mistake.)

## Organization
In the next section you will find how to install and run the program. Beyond that point, we wil organize the project by weeks, so we will explain what is added each week, give some insight on our decision for some implementations and give some final results we got to evaluate our algorithm.

## Installation
This project has been developed using Python 3.13.7, but any Python version above Python 3 should work.

To clone this repo you can do:

```bash
git clone https://github.com/MCV-2025-C1-Project/Team1.git
```


To install the used libraries run the following command:

```bash
cd Team1
pip install -r requirements.txt
```

And you are good to go!

## Usage

To run the program, you have two options.

The first one and more straightforward is to use one of the provided .yaml files inside the ```configs/``` folder. But before that remember to change the path to your database and query set in the .yaml files. Then you can run the program like this:

```bash
python main.py --config configs/test1.yaml
```

The second one is fine to run the program just once, but if you want to rerun a configuration we don't recommend using this. Instead of specifying a .yaml file, you should add the arguments when running the script like so:

```bash
python main.py --database_path <path/to/your/database> --query_path <path/to/your/query/set> --k 1
```

There are more options to play with the color spaces and distances used in case you want to try it out. To get more information about what each argument does run:
```bash
python main.py --help
```

## Week 1
### Development
In this first week we were asked to create an image descriptor and play with different options to choose the one that works best with the model. For that we used the 1D histograms concatenated, as suggested by our teachers, and tried different color spaces to choose the one that fits best.

We also tried different distances to see which are the best ones. But we aren't limited to use just one distance for the final chosen distance, so we decided to try the N best metrics as an ensemble of distances!

### Results


## Team Members
This is the awesome team who collaborated in this project:
* adriangt2001
* MaiolSabater
* GerardVilaplana
* juliagartor
