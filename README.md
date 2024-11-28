# FLARE: Towards Flexible and Lightweight Learning-Based Indexing for High-Dimensional Approximate Nearest Neighbor Search


## Dependency

+ libtorch 2.4
+ json : https://github.com/nlohmann/json
+ stb_image : https://github.com/nothings/stb/tree/master
+ cmake 3.28
+ g++ version: 13

## Running
### Step 1: Path Configuration
+ You need to configure `utils/dataset.hpp::source_path` and `utils/dataset.hpp::father_path`. These two parameters represent the location of the raw dataset and the preprocessed dataset, respectively.
+ If you have a GPU device, you need to configure `utils/Model.hpp::CPU_DEVICE` to enable neural network training on the GPU.

### Step 2: Data Preprocessing

You can use the function from the `utils/TensorStorage.hpp::ReadSourceFile` class to convert the dataset into a Tensor-specific format. In our code, it is represented as a `.tensor` file for fast data loading.

### Step 3: Generate Executable File

FLARE is the version that requires storing data in the VP+Tree,  
FLAREFast is the version that does not require storing data in the VP+Tree.

You can directly use the default cmake command:
``cmake .. && make affix_fast``


### Step 4: Run FLARE

When running the program, it will load the default dataset, default data size, and default k value in the `main` function. To modify these, you can use the following command:
``./affix_fast --k [1,20,100] --datasize 1000000 --dataset TinyImages``


## Key Files and Directories

### `utils/Workload.hpp`
+ **TensorStorage::readTensor**: Loads a tensor from a file into the `torch::Tensor` format, which is used as the dataset `D`.
+ **Workload**: Responsible for loading data, generating query sets, and ground truth.

### `indexes/VPPLUS.hpp`
+ **VPPLUS:VP+Tree**

### `indexes/VPPLUSS.hpp`
+ **VPPLUS:VP+Tree without storaging the data points in leaf nodes**

### `utils/Network.hpp`
+ **FC_NET:fully connected network with multi hidden layers**

### `FLAREFast.hpp`
+ **FLAREFast:FLARE index**

### `runFLAREfast.cpp`
+ **The example file for running FLARE**

