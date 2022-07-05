## SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks

Xiang Xu, Karl D.D. Willis, Joseph G. Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, Yasutaka Furukawa

**ICML 2022**  
[website](https://samxuxiang.github.io/skexgen/index.html) | [paper]() | [video]()


## Installation

### Environment
* **Tested OS:** Linux
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
Note this is only tested on CUDA 11.4 and up. 

 
## Data

### Preprocess
Download original DeepCAD json from [here](https://github.com/ChrisWu1997/DeepCAD) ([Google Drive](https://drive.google.com/drive/folders/1mSJBZjKC-Z5I7pLPTgb4b5ZP-Y6itvGG)) and put it under `data` folder.

Follow these steps to convert DeepCAD data to SkexGen format (under `utils` folder):
```bash
# parse DeepCAD json to a simple obj format 
  python convert.py --data_folder ../data/cad_json --output_folder ../data/cad_obj

# normalize CAD and update the obj file
  python normalize.py --data_folder ../data/cad_obj --out_folder ../data/cad_norm

# parse obj to primitive sequence 
  python parse.py --input ../data/cad_norm --output ../data/cad_data --bit 6

# remove sketch training data duplicates
  python deduplicate.py --datapath path/to/cad_network --hash_type s

# remove extrude training data duplicates 
  python deduplicate.py --datapath path/to/cad_network --hash_type e
```
### Pretrained Models
Download pretrained SkexGen model from [here]()


## Training

To train the sketch module (topology encoder, geometry encoder, sketch decoder):
  ```
    python train_sketch.py --data path/to/cad_network/train_unique_s.pkl \
                      --output proj_log/your/exp \
                      --bit 6 --maxlen 200 --batchsize 128 --device 0
  ```
  `maxlen`: sketch sequence length

To train the extrude module (extrude encoder, extrude decoder):
  ```
    python train_extrude.py --data path/to/cad_network/train_unique_e.pkl \
                      --output proj_log/your/exp \
                      --bit 6 --maxlen 5 --batchsize 128 --device 0
  ```
  `maxlen`: number of extudes, extrude sequence length is `maxlen` x 20


Extract training dataset codes:
  ```
    python extract_code.py --weight proj_log/your/exp \
                           --epoch 300 --device 0 --maxlen 200 --bit 6 \
                           --output proj_log/your/exp/codes \
                           --data path/to/cad_network/train_unique_s.pkl \
                           --invalid path/to/cad_network/train_invalid_s.pkl 
  ```

To train the code selector: 
  ```
    python train_code.py --input proj_log/your/exp/codes/train_code.pkl \
                       --output proj_log/your/exp \
                       --batchsize 512 --device 0 \
                       --code 1000 --seqlen 10
  ```
  `seqlen`: 4 topology codes, 2 geometry codes, 4 extrude codes 
  `code`: max size of codebook is 1000



## Testing and Evaluation
Randomly sample the codes and decode to sketch-and-extrude: 
  ```
    python sample.py --weight proj_log/your/exp \
                        --epoch 300 --device 0 --maxlen 250 --bit 6 \
                        --output proj_log/your/exp/samples \
                        --data path/to/cad_network/train_unique_s.pkl \
                        --invalid path/to/cad_network/train_invalid_s.pkl 
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
