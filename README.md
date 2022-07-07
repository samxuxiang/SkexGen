## SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks

Xiang Xu, Karl D.D. Willis, Joseph G. Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, Yasutaka Furukawa

**ICML 2022**  
[project](https://samxuxiang.github.io/skexgen/index.html) | [paper]() | [youtube]() | [bilibili]()


## Installation

### Environment
* Linux
* Python >= 3.8
* PyTorch >= 1.10

### Dependencies
* Install [PyTorch 1.10](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
* Install other dependencies:
    ```
    pip install -r requirements.txt
    ```
* Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core).

### Docker
We also provide the docker image for running SkexGen. You can download it from [dockerhub](https://hub.docker.com/r/samxuxiang/skexgen) (~10GB). \
Note: only tested on CUDA 11.4. 

 
## Data

Download the [raw json data](https://drive.google.com/drive/folders/1mSJBZjKC-Z5I7pLPTgb4b5ZP-Y6itvGG) from [DeepCAD](https://github.com/ChrisWu1997/DeepCAD). Unzip it under the data folder.

Follow these steps to convert DeepCAD data to SkexGen format:
```bash
# Under utils folder:
-----------------------------------Old--------------------------
# parse DeepCAD json to a simple obj format 
  python convert.py --data_folder ../data/cad_json --output_folder ../data/cad_obj

# parse obj to network-friendly sequence and save as pickle (under `data_utils` folder)
  python parse.py \
    --input path/to/cad_norm \
    --train_val_test_split path/to/deepcad/train_val_test_split.json \
    --output path/to/cad_network \
    --bit 6

# remove sketch training data duplicates (under `data_utils` folder)
  python deduplicate.py \
    --data_folder path/to/cad_network \
    --train_val_test_split path/to/deepcad/train_val_test_split.json \
    --hash_type 's' 

# remove extrude training data duplicates (under `data_utils` folder)
  python deduplicate.py \
    --data_folder path/to/cad_network \
    --train_val_test_split path/to/deepcad/train_val_test_split.json \
    --hash_type 'e' 

# create the invalid file list (under `data_utils`)
  python invalid.py --datapath path/to/cad_network/ --bit 6

```

When running `convert.py` some files in the DeepCAD dataset fail to generate valid solid models.  You may use the the `--verbose` option to see additional details about the problem files.   If the `convert.py` script hangs during processing it can be safely restarted and will continue from where it left off.



```
-----------------------------------New--------------------------
# normalize CAD and update the obj file
  python normalize.py --data_folder ../data/cad_obj --out_folder ../data/cad_norm

# parse obj to primitive sequence 
  python parse.py --input ../data/cad_norm --output ../data/cad_data --bit 6

# remove duplicated sketch data
  python deduplicate.py --datapath ../data/cad_data --hash_type s

# remove duplicated extrude data
  python deduplicate.py --datapath ../data/cad_data --hash_type e

# Find all the invalid CAD models
  python invalid.py --datapath ../data/cad_data --bit 6
```

You can download the already [pre-processed data](https://drive.google.com/file/d/1so_CCGLIhqGEDQxMoiR--A4CQk4MjuOp/view?usp=sharing)



## Training

Train sketch branch (topology encoder, geometry encoder, sketch decoder):
  ```
    python train_sketch.py --data data/cad_data/train_deduplicate_s.pkl \
                           --output proj_log/exp_sketch \
                           --invalid data/cad_data/train_invalid.pkl \
                           --bit 6 --maxlen 200 --batchsize 128 --device 0
  ```
  `maxlen`: sketch sequence length (default 200)

Train extrude branch (extrude encoder, extrude decoder):
  ```
    python train_extrude.py --data data/cad_data/train_deduplicate_e.pkl \
                            --output proj_log/exp_extrude \
                            --bit 6 --maxlen 5 --batchsize 128 --device 0
  ```
  `maxlen`: number of extudes (default 5)


Extract codes:
  ```
    python extract_code.py --sketch_weight proj_log/exp_sketch \
                           --ext_weight proj_log/exp_extrude \
                           --device 0 --maxlen 200 --bit 6 \
                           --output proj_log/exp_code \
                           --data data/cad_data/train.pkl \
                           --invalid data/cad_data/train_invalid.pkl 
  ```

Train code selector (random generation): 
  ```
    python train_code.py --input proj_log/exp_code/code.pkl \
                         --output proj_log/exp_code \
                         --batchsize 512 --device 0 \
                         --code 1000 --seqlen 10
  ```
  `seqlen`: 4 topology, 2 geometry, 4 extrude, 
  `code`: max size of codebook is 1000

Download our [pretrained models](https://drive.google.com/file/d/1K4zxfoL7W9Q--d8wVv4spCf4ARVNKxqK/view?usp=sharing)


## Evaluation
Random generation: 
```bash
# sample the codes and autoregressively decode it to sketch and extrude
  python sample.py --sketch_weight proj_log/exp_sketch \
                      --ext_weight proj_log/exp_extrude \
                      --code_weight proj_log/exp_code \
                      --device 1 --bit 6 \
                      --output proj_log/samples 
```

Visualization: 
```bash
# Under utils folder:

# convert generated sketch-and-extrude to stl format (timeout prevent occ hanging)
  timeout 180 python visual_obj.py --data_folder ../proj_log/samples 

# render and visualize to images 
  python cad_img.py  --input_dir ../proj_log/samples --output_dir ../proj_log/samples_visual
```
                

Evaluate the CAD models (after running `visual_obj.py`):
```bash
# Under utils folder:

# uniformly sample 2000 points 
  python sample_points.py --in_dir ../proj_log/samples --out_dir pcd

# evaluate performance 
  python eval_cad.py --fake ../proj_log/samples \
                     --real ../data/test_eval
```
Download [test_eval](https://drive.google.com/file/d/1R_Tzourk3XDIDUsnTn_UJVq3uVWe5s38/view?usp=sharing) and unzip it under the data folder. This contains the point clouds from DeepCAD test set. 

## Citation
If you find our work useful in your research, please cite our paper [SkexGen](https://samxuxiang.github.io/skexgen):
```
@inproceedings{ICML,
  title={SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks},
  author={xxx},
  year={2022}
}
```

## License
Please see the [license](LICENSE) for further details.

---
**Update (07/05/2022)**: Full code released.\
**Update (06/06/2022)**: Evaluation code added.\
**Update (06/05/2022)**: Training code added.\
**Update (05/30/2022)**: Code will be released soon!


