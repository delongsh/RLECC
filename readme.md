This repository is code for our paper **Face Template Protection Through Residual Learning Based Error-Correcting Codes**.
## Introduction
This code aims to implement a face template protection technique by using residual learning, which maps the facial images into low-density parity-check (LDPC) codewords. This implementation is based on [PyTorch][1]. In this paper, we use the [LDPC coding][2] algorithm developed by Radford M. Neal. The [trained CNN model on extended Yale Face B with d = 256][3] can be downloaded.


## Directory
### Root
```
${ROOT}
|-- labels
|-- load_datasets
|-- models
|-- process_res
|-- split
|-- test_images
main-test.py
utils.py
requirements.txt
```
 - labels: LDPC codewords for different face databases
 - load_datasets: Load face images
 - models: Define CNN architecture
 - process_res: Process the predicted binary codes
 - split:  Store different orders for face database
 - test_images: Store test images, [download][4] .
## Database
The databases used in our paper contain [extended Yale Face B][5], [PIE][6] and [FEI][7] Database.

## Requirements
Install python packages
```bash
pip install -r requirements.txt
```
## Running
```bash
python main-test.py -r 11-04-13-20 --seed 1 --dataset yaleB --ldpc-len 256 --dataset-seed 0 -t 38 0 --epochs 50 --batch-size 256
```
## More Info

 - **Contact**: Please send comments to <shang_delong@163.com>

  [1]: https://pytorch.org/
  [2]: http://www.cs.utoronto.ca/~radford/ldpc.software.html
  [3]: https://drive.google.com/drive/folders/1Hy92DZte4i7HyC07FASn9vSwqY78_Elv?usp=sharing
  [4]: https://drive.google.com/drive/folders/12s7cnE9VSPLyEGCgFssgMGGP2VmemvRR?usp=sharing
  [5]: http://vision.ucsd.edu/content/extended-yale-face-database-b-b
  [6]: http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html
  [7]: https://fei.edu.br/~cet/facedatabase.html
