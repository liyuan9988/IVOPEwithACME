# Walkthrough for running the code
The following is tested on Ubuntu 16.

1. Clone the repository
   
   ```
   git clone https://github.com/liyuan9988/IVOPEwithACME.git
   cd IVOPEwithACME
   ```

2. Set up python environment with `pyenv` and all dependencies
   
   ```
   PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.9
   pyenv local 3.7.9
   pip install -r requirements.txt
   ```

3. Download Training Data from Google Cloud Storage

    ```
    gsutil -m cp -r  gs://rl_unplugged/bsuite/ ./
    gsutil -m cp -r  gs://rl_unplugged/bsuite_near_policy/ ./
    gsutil -m cp -r  gs://rl_unplugged/dm_control_suite_stochastic/ ./
    ```

4. Run scripts `main/*.sh` after altering `DATA_PATH`

