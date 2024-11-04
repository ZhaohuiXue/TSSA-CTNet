# TSSA-CTNet

This repository includes codes and dataset for "Temporal-spectral-semantic-aware convolutional transformer network for multi-class tidal wetland change detection in Greater Bay Area", which has been published in ISPRS Journal of Photogrammetry and Remote Sensing.    

The materials in this repository are only for study and research, NOT FOR COMMERCIAL USE. **Please cite this paper if it is helpful for you**.
***

### Requirements： 
 ```
　cuda 11.7  
　numpy 1.24.1 
　python 3.610  
　pytorch 2.0.0  
  ```

### Data preparation  
The shape of the input data :  (B, C, T, H, W)
  B is the Batch size 
  C is the Channels  
  T is the length of Time  
　H is the High of the image patch  
　W is the Width of the image patch  
 
### Train
python train.py  
Note there is a config file named second.yaml  
