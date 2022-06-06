# SkexGen: Generating CAD Construction Sequences by Autoregressive VAE with Disentangled Codebooks

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
We also provide the docker image for running SkexGen. You can download it from [dockerhub](https://hub.docker.com/r/samxuxiang/skexgen) (~10GB). 

Note this is only tested on CUDA 11.4 and up. 

 
## Data
* Download original DeepCAD json data from [here](https://github.com/ChrisWu1997/DeepCAD).
* Go inside `occ_utils` folder.
* Convert json to obj format and also save its stl
  ```
    python convert.py --data_folder path/to/cad_json --output_folder path/to/cad_obj
  ```
* Normalize CAD  
  ```
    python normalize.py --data_folder path/to/cad_obj --out_folder path/to/cad_norm
  ```
* Go inside `data_utils` folder.
* Parse obj to network-friendly sequence and save as pickle
  ```
    python parse.py --input path/to/cad_norm --output path/to/cad_network --bit 6
  ```
* Remove duplicates 
  ```
    python deduplicate.py --datapath path/to/cad_network --hash_type 's'
  ```
  hash_type: `s` for sketch training and `e` for extrude training.

## Training
* Training sketch module (topology encoder, geometry encoder, sketch decoder)
```
  python train_s.py --data path/to/cad_network/train_unique_s.pkl \
                    --output proj_log/your/exp \
                    --bit 6 --maxlen 250 --batchsize 256 --device '0' 
```

## Evaluation


## Citation
If you find our work useful in your research, please cite our paper [SkexGen](https://samxuxiang.github.io/skexgen):
```
@inproceedings{xxx,
  title={SkexGen: Generating CAD Construction Sequences by Autoregressive VAE with Disentangled Codebooks},
  author={xxx},
  booktitle={xxx},
  year={xxx}
}
```

## License
Please see the [license](LICENSE) for further details.

---
**Update (06/06/2022)**: Training and evaluation added.
**Update (05/30/2022)**: Code will be released soon!
