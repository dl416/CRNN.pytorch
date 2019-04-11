# CRNN.pytorch

## 0. Prepare

This code tested on ubuntu 18.04.
First, you should install wrap_ctc used
```
pip install pip install torch-baidu-ctc
```
Reference: [wrap-ctc](https://github.com/jpuigcerver/pytorch-baidu-ctc)

Second, you should Download data in a directory named data.
Files are organized like follow.

```
./
  -- data/
      -- IIIT5K
  -- model/
  -- source/
```

### 1. Train

```
python ./source/train.py
```

