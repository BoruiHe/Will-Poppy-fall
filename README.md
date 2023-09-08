### Introduction
This repo is the implementation of the paper [Understanding Hyperdimensional Computing for Parallel Single-Pass Learning](https://arxiv.org/abs/2202.04805).

### Usage

1. Download the real Poppy dataset from this [link](https://drive.google.com/file/d/1gnMWdRNPNHEHCsTN40nOruPfZXFxTSHg/view?usp=drive_link).
2. Generate your virtual Poppy dataset by executing data_generation.py.
```
python data_generation.py [-gs gs] [-n n] [-tp type] 
optional arguments:
  -gs GS                global scale
  -n N                  the number of videos of each case you want every time the script is executed
  -tp TYPE              'fall' for N falls videos, 'standing' for N standing videos, 'both' for N fall videos and N standing videos
```
Foe example,
```
python data_generation.py -gs 10 -n 1 -type both
```
will generate 1 falling videos and 1 standing videos. You can find them in the `virtual_poppy` folder.
3. For testing the 3rd party model, execute baseline.py.
4. For all experimental results, execute main.py. (You have to set hyperparameters for your personal configuration.)
5. example_gaga.py will show you an example of image reconstructions.
### Citation
If you find this repo useful, please cite:
