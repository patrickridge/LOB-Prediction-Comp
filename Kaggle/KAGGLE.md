## Set Up
 - Create Notebook
 - Make sure done phone verification in settings
 - In notebook -> settings -> accelerator -> GPU T4x2
 - copy cells from github kaggle/kaggle_train.ipynb 
 
 # Data and Files
 - Right side of new notebook
 - Upload -> new dataset -> import train.parquet and valid.parquet
 - Upload -> new model -> train.py and solution.py

# Github
- kaggle/train_gru.py
        /solution.py
        /utils.py

check paths are correct in notebook

## Uploading
- once .pt file saved to kaggle/working
- dowload best version
- import .pt file into cursor into artifacts folder
- change the following for current test in solution.py:

CKPT_NAME = "gru_best_h128_L6_do0.1.pt"

DEFAULT_INPUT_DIM = 32
DEFAULT_D_OUT = 2
DEFAULT_HIDDEN = 128
DEFAULT_NUM_LAYERS = 6
DEFAULT_DROPOUT = 0.1

then zip file as normal and ready to submit