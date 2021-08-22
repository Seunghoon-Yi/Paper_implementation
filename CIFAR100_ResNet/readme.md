# Practice 1 : CIFAR-100 dataset with ResNet
<br/>

## Environment : 
<br/>

    python                    3.9.6
    pytorch                   1.9.0
    cudatoolkit               11.1.1
    cudnn                     8.2.1

<br/>
<br/>

## Description

### train.py
<br/>

**1. Required options and arguments**<br/> 

        '--data_dir', default='./data/train'
        '--model', the model name to train. ResNet 18/34/50/101/112 are implemented.
        '--lr', initial learning rate. default = 0.001
        '--epoch', default=50
        '--BS', batch size. default=64.
        '--gpu',  default='False', true when gpu is available.
        
<br>

**2. Example :**
<br>
        $python train.py --model ResNet34 --epoch 20 --lr 0.009 --BS 128 --gpu
        
output : Accuracy and trained model. After training procedure, the best model will be stored in <code>./results/model_name/</code>
