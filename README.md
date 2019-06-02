# BCNN
This project makes a simple BCNNN example. In the "BCNN_VGG16.py", I construct the BCNN based on VGG16 on keras. Meanwhile,  I add a "dimension reduction layer" to the BCNN, which can be seen in "BCNN_Advanced.py". By using "convolutional dimision reduction", the parameters of BCNN become less than before, so I call the new model "A-BCNN".
# train or test
If you want to train the model, you can use the command like this:`python BCNN_VGG16.py --train`  
If you want to test the model, you can use the command like this:  `python BCNN_VGG16.py --test`  
There are three command parameters: `--classes --num --lr`
- --classes: the num of classes. you can ignore it ,this para is set for my own dataset. You can change this in the main func.
- --num: the num of freezed layers. If you set  `--num -1`, then layes[0:-1] will be set notrainable.
- --lr: learning rate
# others
随便做的，有时间再改吧
