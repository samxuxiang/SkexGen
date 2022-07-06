## SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks

Xiang Xu, Karl D.D. Willis, Joseph G. Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, Yasutaka Furukawa

**ICML 2022**  
[website](https://samxuxiang.github.io/skexgen/index.html) | [paper]() | [video]()


## Installation

### Environment
* Linux
* Python >= 3.8
* PyTorch >= 1.10.1

### Dependencies
* Install [PyTorch 1.10.1](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
* Install other dependencies:
    ```
    pip install -r requirements.txt
    ```
* Install pythonocc following the instruction [here](https://github.com/tpaviot/pythonocc-core).

### Docker
We also provide the docker image for running SkexGen. You can download it from [dockerhub](https://hub.docker.com/r/samxuxiang/skexgen) (~10GB). \
Note: only tested on CUDA 11.4. 

 
## Data

### Preprocess
Download original DeepCAD json [here](https://github.com/ChrisWu1997/DeepCAD) ([Google Drive](https://drive.google.com/drive/folders/1mSJBZjKC-Z5I7pLPTgb4b5ZP-Y6itvGG)) and put it under `data` folder.

Follow these steps to convert DeepCAD data to SkexGen format:
```bash
# Under utils folder:

# parse DeepCAD json to a simple obj format 
  python convert.py --data_folder ../data/cad_json --output_folder ../data/cad_obj

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

Download our pre-processed SkexGen data [here](https://drive.google.com/file/d/1so_CCGLIhqGEDQxMoiR--A4CQk4MjuOp/view?usp=sharing)



## Training

Train sketch module (topology encoder, geometry encoder, sketch decoder):
  ```
    python train_sketch.py --data data/cad_data/train_deduplicate_s.pkl \
                           --output proj_log/exp_sketch \
                           --invalid data/cad_data/train_invalid.pkl \
                           --bit 6 --maxlen 200 --batchsize 128 --device 0
  ```
  `maxlen`: sketch sequence length (default 200)

Train extrude module (extrude encoder, extrude decoder):
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
                           --data data/cad_data/train_deduplicate_s.pkl \
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

Download our pretrained SkexGen model [here]()


## Testing and Evaluation
Randomly sample the codes and decode to sketch-and-extrude: 
  ```
    python sample.py --sketch_weight proj_log/exp_sketch \
                     --ext_weight proj_log/exp_extrude \
                     --code_weight proj_log/exp_code \
                     --device 0 --bit 6 \
                     --output proj_log/samples 
  ```                   

Evaluate the results by COV, MMD and JSD:
```bash
# (under utils folder)
# convert generated sketch-and-extrude to stl 
  python visual_obj.py --data_folder proj_log/your/exp/samples

# uniformly sample 2000 points on the CAD model
  python sample_points.py --in_dir proj_log/your/exp/samples --out_dir pcd

# evaluate generation performance 
  python eval_cad.py --fake proj_log/your/exp/samples \
                     --real path/to/cad_network/test_obj
```


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
