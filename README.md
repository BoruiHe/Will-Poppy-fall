### Introduction
This repo is the implementation of the paper [Will Poppy Fall? Predicting Robot Falls in Advance Based on Visual Input](https://ieeexplore.ieee.org/document/10459748).

### Usage
1. Download the real Poppy dataset from this [link](https://drive.google.com/file/d/1gnMWdRNPNHEHCsTN40nOruPfZXFxTSHg/view?usp=drive_link) and unzip it to the `Will-Poppy-fall-main` folder.
2. Download the 3rd part dataset from [here](https://doi.org/10.34894/3DV8BF).
3. Generate your virtual Poppy dataset by executing data_generation.py. Any modification on `global scale` is not recommended.
```
python data_generation.py [-gs gs] [-n n] [-tp type] 
optional arguments:
  -gs GS                global scale
  -n N                  the number of videos of each case you want every time the script is executed
  -tp TYPE              'fall' for N falls videos, 'standing' for N standing videos, 'both' for N fall videos and N standing videos
```
For example,
```
python data_generation.py -gs 10 -n 1 -type both
```
will generate 1 falling video and 1 standing video. You can find them in the `virtual_poppy` folder.

4. Execute preprocessing.py to generate a `videos.h5` file.

For example, run
```
python utils/preprocessing.py
```
on Windows OS.

5. You may need to create the `checkpoints` folder before run any experiment.
6. Execute baseline.py for training and testing the 3rd party model.
7. For all experimental results, execute main.py. (You have to set hyperparameters for your personal configuration.)
8. example_gaga.py will show you an example of image reconstruction.

<!-- ### Citation
If you find this repo useful, please cite: -->
