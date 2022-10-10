1. Clone the project
    ```
    git clone https://github.com/x51ming/rep-attention.git
    cd rep-attention
    ```

2. Create a conda environment
    ```
    conda create -n rep-attention python=3.8
    conda activate rep-attention
    ```

3. Install pytorch, torchvision and cudatoolkit
    ```
    conda install pytorch==1.8.2 torchvision==0.9.2 cudatoolkit=11.1 -c pytorch-lts -c nvidia
    ```

4. Install required packages
    ```
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
    pip install decorator tensorboardX rich
    ```

5. Install APEX (optional, for mixed precision training)
    ```
    git clone https://github.com/NVIDIA/apex
    pushd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    popd
    ```

6. Install lmdb
    ```
    pip install lmdb
    ```

    