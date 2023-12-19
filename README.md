# Pose Classification with 3D Atomic Neural Network 2 (PECAN2)

Identifying correct binding poses of ligands is important in docking based virtual high-throughput screening. This code implements two convolutional neural network approaches: a 3D convolutional neural network (3D-CNN) and a point cloud network (PCN) to improves virtual high-throughput screening to identify novel molecules against each target protein. The code is written in python with Pytorch.


## Prerequsites
- [PyTorch](https://pytorch.org)
- [Open Drug Discovery Tool Kit (ODDT)](https://oddt.readthedocs.io/en/latest/)
- [Open Babel](https://openbabel.org/docs/dev/Installation/install.html)
- [RDkit](https://www.rdkit.org)


## Running the application

### Data Format
PCN use a 3D atomic representation as input data in a Hierarchical Data Format (HDF5). See (https://github.com/LLNL/FAST/) for more information about this HDF5 format.

### PCN
To train, ```PCN_main_train.py``` To test/evaluate, run ```PCN_main_eval.py```. Here is an example comand to evaluate a pre-trained PCN model:
```
python PCN_main_eval.py --device-name cuda:1  --data-dir /Data   --mlhdf-fn pdbbind2019_core_docking.hdf  --csv-fn vina_delta.csv   --model-path /Model_Checkpoint/PCN_a.pth
```


## Authors

PECAN was created by Heesung Shim (shim2@llnl.gov)

## License

PECAN is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.
