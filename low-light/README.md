#### About Traditional Method:

run example:

```
# python3 demo.py -i "images/test_001.jpg" -s "results/test_001_he.jpg" -m msr
```



#### About ML Model:

- create conda environment[optional]

```
conda create -n zerodce_env python=3.8
conda activate zerodce_env
```

- run test 

```python
python lowlight_test.py
```

- Run training

```python
python lowlight_train.py 
```