# SLT Transformer #
<br>

## 1. Data preprocessing ##
<br>

This model is trained on PHOENIX-14T sign language dataset, which videos has size of [T, H, W] = [T, 260, 210].
<br>
**1)** 
Video frames are padded to T = 224, and center-cropped with size of 228, 196.<br>
**2)** 
Then saved into a predefined directory. This process is done by executing 

```save_video_multiprocessing.py```.


<br><br>

## n. Sample Results ##
<br>
GT sentences of test dataset and its inference sentences from the model are given in the 

**Sample results**

directory. BLEU score model is also included. 
