# A PyTorch implementation of CRNN

##  CTC

 - torch.nn.CTCLoss()
 - warpctc_pytorch.CTCLoss()
   - [Github](https://github.com/SeanNaren/warp-ctc)

## Compare torch.nn.CTCLoss & warpctc_pytorch.CTCLoss


**Note: by default torch.nn.CTCLoss: reduction = 'mean' which means you should set size_average=True in warpctc_pytorch.CTCLoss**




|CTC Method|preds.device|labels.device|cudnn|result|
|:-:|:-:|:-:|:-:|:-:|
|nn.CTCLoss|cuda|cuda|False|normal|
|nn.CTCLoss|cuda|cuda|True|cudnn_ctc_loss error|
|nn.CTCLoss|cuda|cpu|True|NaN|
|nn.CTCLoss|cpu|cpu|True/False|normal but slowly|
|warpctc.CTCLoss|cuda|cuda|True/False|segmentation fault|
|warpctc.CTCLoss|cuda|cpu|True/False|normal|
|warpctc.CTCLoss|cpu|cpu|True/False|normal|

