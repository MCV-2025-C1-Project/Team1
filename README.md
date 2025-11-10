# C1 Project
Welcome to the Team1's project of the C1 course from the MCV 25-26!

With this project, given a certain database of images (i.e. art from a museum) you can retrieve information of each element of the database with another image taken by you. This technique is called Content Based Image Retrieval (CBIR), and our best method uses feature extraction and matching algorithms. It currently relies on masking the image to extract every individual painting, but it should also work with no masked images. Images with no features (plain colors or very blurry images) may fail in retrieving the correct painting from the database.

## Organization
In the next section you will find how to install and run the program. Beyond that point, we wil organize the project by weeks, so we will explain what is added each week, give some insight on our decision for some implementations and give some final results we got to evaluate our algorithm.

## Installation
This project has been developed using Python 3.13.7, but any Python version above Python 3 should work.

To clone this repo you can do:

```bash
git clone https://github.com/MCV-2025-C1-Project/Team1.git && cd Team1
```

We recommend installing the required libraries inside a virtual environment:

Create and activate the virtual environment:

Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux or macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```


And you are good to go!

## Usage

### Week 1
To run the program, you have two options.

The first one and more straightforward is to use one of the provided .yaml files inside the ```configs/``` folder. But before that remember to change the path to your database and query set in the .yaml files. Then you can run the program like this:

```bash
python week1.py --config configs/test.yaml
```

The second one is fine to run the program just once, but if you want to rerun a configuration we don't recommend using this. Instead of specifying a .yaml file, you should add the arguments when running the script like so:

```bash
python week1.py --database_path <path/to/your/database> --query_path <path/to/your/query/set> --k 1
```

There are more options to play with the color spaces and distances used in case you want to try it out. To get more information about what each argument does run:
```bash
python week1.py --help
```

### Week 2

You can run the program in two ways: using a YAML config (recommended) or passing arguments directly on the command line. 
1) Run with a YAML config: Put a config file inside configs/ (or anywhere you like) and set at least your database and query paths. Then run:
   ```bash
    python week2.py --config configs/experiment.yaml
    ```
2) Run with CLI arguments (quick one-off):
    ```bash
    python week2.py \
    --database_path /path/to/database \
    --query_path /path/to/queries \
    --color_space_list lab \
    --preprocesses_list None \
    --bins_list 64 \
    --blocks_list 1 \
    --hist_dims_list 1 \
    --distances_list hist_intersection \
    --k_list 1 5 10
    ```
In case you need help to execute or check the default values run:
```bash
python week2.py --help
```

### Week 3
Similar to week 2, you can run the program in two ways: using a YAML config (recommended, more clean) or passing arguments directly on the command line. 
1) Run with a YAML config: Put a config file inside configs/ (or anywhere you like) and set at least your database and query paths. Then run:
   ```bash
    python week3.py --config configs/experiment.yaml
    ```
   An example of a .yaml file could be:
   ```yaml
   data:
     database_path: datasets/BBDD/
     query_path: datasets/qst1_w3/
   
   retrieval:
     color_space_list: [gray_scale]
     preprocesses_list: ['']
     masking: false
     denoising: true
   
     descriptors_list: [DCT]
     
     blocks_list: [4]
     DCT_coeffs_list: [16]
     
     distances_list: [canberra]
     k_list: [10]
   
   evaluation:
     val: false
   
   output:
     pickle_filename: 'results/w3/w3_sample_1.pkl'
   ```

2) Run with CLI arguments (quick one-off):
   ```bash
   python week3.py \
    --database_path ./data/db \
    --query_path ./data/queries \
    --color_space_list lab rgb \
    --preprocesses_list None clahe \
    --masking True \
    --denoising False \
    --descriptors_list hist LBP \
    --bins_list 32 64 \
    --blocks_list 1 4 \
    --hist_dims_list 1 \
    --LBP_scales_list "(8,1.0)" "(16,2.0)" \
    --OCLBP_uniform_u2 True \
    --DCT_coeffs_list 16 32 \
    --wavelets_list bior1.1 haar \
    --distances_list hist_intersection euclidean \
    --k_list 1 5 10 \
    --val True \
    --pickle_filename results.pkl
   ```

### Week 4
Keeping the things aligned with the previous week, this week's script also offers the 2 options to be executed, either by providing a yaml file with the configuration (recommended since it stays more clean) or by providing the arguments directly to the terminal.

1) Run with a YAML config: Put a config file inside configs/ (or anywhere you like) and set at least your database and query paths. Then run:
   ```bash
    python week4.py --config configs/experiment.yaml
    ```
   An example of a .yaml file for sift descriptors could be:
   ```yaml
   # Rutas
   database: "../BBDD"
   dataset: "../qsd1_w4"

   kp_descriptor: ["sift"]
   
   # SIFT
   n_features: [500]
   edge_threshold: [10]
   n_octave_layers: [4]
   contrast_threshold: [0.02]
   sigma: [1.6]
   
   # ORB
   scale_factor: [1.2]
   n_levels: [8]
   first_level: [0]
   WTA_K: [3]
   patch_size: [31]
   fast_threshold: [20]
   
   # MAP@K a evaluar
   k: [1,5]
   
   # Modo y salida
   mode: "search"
   output_csv: "./grid_search_sift.csv"
   ```
2) Run with CLI arguments (quick one-off):
   ```bash
   python week4.py \
    --database ./data/database \
    --dataset ./data/queries \
    --kp_descriptor SIFT ORB \
    --n_features 500 1000 \
    --edge_threshold 10 15 \
    --n_octave_layers 3 4 \
    --contrast_threshold 0.04 0.03 \
    --sigma 1.6 1.2 \
    --scale_factor 1.2 1.5 \
    --n_levels 8 10 \
    --first_level 0 \
    --WTA_K 2 \
    --score_type 0 \
    --patch_size 31 \
    --fast_threshold 20 \
    --descriptor_size 128 64 \
    --descriptor_channels 3 \
    --threshold 0.001 0.005 \
    --n_octaves 4 5 \
    -k 1 5 10 \
    --mode eval \
    --output_pkl sift_orb_results.pkl \
    --output_csv sift_orb_results.csv
   ```
## Week 1

During the first week, our objective was to design an image descriptor and explore different configurations to determine the most effective approach for the painting similarity matching task. Following our instructors’ guidelines, we built descriptors based on concatenated 1D color histograms and evaluated multiple color spaces, numbers of bins, distance metrics, and preprocessing strategies. To ensure a fair and comprehensive comparison, we conducted a grid search over all these parameters to identify the configuration that delivered the most reliable and discriminative results.

## Week 2

During the second week, our objective was to implement a new method for computing histograms by dividing images into blocks to calculate local histograms. Additionally, we received a new set of images that included backgrounds, and our task was to develop a method for background removal. Our approach consisted of estimating the average color of the background, isolating the largest connected component, filling the gaps within that component, and finally identifying the largest square region corresponding to the object of interest. With the background successfully removed using this mask, we were then able to process the images with backgrounds through the updated retrieval system.

## Week 3

Our first task was to implement a noise removal system and evaluate it using two different metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).After testing multiple algorithms and configurations, we obtained the following results:

Best PSNR configuration:
- Algorithm: Bilateral Filter
   - Kernel Size: 3
   - Sigma: 1
   PSNR: 52.14
   SSIM: 0.8628

Best SSIM configuration:
- Algorithm: Median Filter
   - Kernel Size: 3
   - PSNR: 33.33
   - SSIM: 0.9005

Once we identified the best denoising setup, we applied it to preprocess our images before using them in the texture-based image retrieval system developed in previous weeks.After extensive testing, we concluded that the best texture descriptor for our painting similarity matching task is the DCT descriptor combined with the Canberra distance metric.

The second task involved implementing a masking algorithm to remove the background from the images, keeping only the paintings — similar to the previous week’s segmentation exercise.A new challenge arose when we discovered that some images contained one painting, while others had two. To handle this, we developed an algorithm capable of distinguishing between these two cases.Using Scharr operators and Hough transformations, we were able to detect and extract the painting regions accurately. Additionally, by analyzing the image gradients, we reduced shadow leakage and refined the mask edges.The final results for the Week 3 masking task on the development set were:

- Precision: 0.97
- Recall: 0.99
- F1-Score: 0.98

Finally, we integrated this background removal algorithm into the image retrieval pipeline. To further improve alignment and matching accuracy, we applied homography transformations to ensure the generated masks matched the orientation of the database images.


## Week 4

For the final week, our main objective was to integrate the image retrieval system with local feature descriptors to enhance the robustness and accuracy of our matching pipeline.

Our first task was to implement and evaluate different local feature descriptors. After extracting the local features from both the query images and the database images, we designed a matching algorithm that compared them to find the most similar paintings. To ensure reliable results, we applied bidirectional matching — verifying that the best matches were consistent in both directions (query → database and database → query). This approach significantly improved the matching precision by filtering out false correspondences.

After comparing all configurations, we concluded that the best-performing local descriptors for our project were ORB and AKAZE: ORB (Oriented FAST and Rotated BRIEF) offered excellent speed, making it highly suitable for real-time or large-scale retrieval. AKAZE provided slightly more accurate feature matches in complex regions but was noticeably slower due to its nonlinear scale-space computation. Ultimately, ORB achieved a much better trade-off between performance and processing time, making it our preferred choice for the final retrieval system.

After fine-tuning all parameters and optimizing the image size, our final system achieved: MAP@1: 0.97.

The presentation slides can be found in the [docs folder](./docs) in pdf and pptx format, and also in this [canva link](https://www.canva.com/design/DAG4BnURqVo/HYiTMUn9OxFn19S1V86y8A/view?utm_content=DAG4BnURqVo&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h985ac2ac9b).

## Team Members
This is the awesome team who collaborated in this project:
* adriangt2001
* MaiolSabater
* GerardVilaplana
* juliagartor
