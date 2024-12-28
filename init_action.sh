#!/bin/bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -euxo pipefail

readonly DEFAULT_TENSORFLOW_VERSION="2.4.1"

TENSORFLOW_VERSION="$(/usr/share/google/get_metadata_value attributes/tensorflow-version || echo ${DEFAULT_TENSORFLOW_VERSION})"
readonly TENSORFLOW_VERSION

INSTALL_TORCHREC="$(/usr/share/google/get_metadata_value attributes/install-torchrec || echo 'true')"
readonly INSTALL_TORCHREC

INSTALL_LLM="$(/usr/share/google/get_metadata_value attributes/install-llm || echo 'false')"
readonly INSTALL_LLM

INSTALL_MERLIN="$(/usr/share/google/get_metadata_value attributes/install-merlin || echo 'false')"
readonly INSTALL_MERLIN

INSTALL_RAPIDS="$(/usr/share/google/get_metadata_value attributes/install-rapids || echo 'true')"
readonly INSTALL_RAPIDS

INSTALL_NLP="$(/usr/share/google/get_metadata_value attributes/install-nlp || echo 'true')"
readonly INSTALL_NLP


function execute_with_retries() {
  local -r cmd=$1
  for ((i = 0; i < 10; i++)); do
    if eval "$cmd"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

function run_multiple_times() {
  local -r cmd=$1
  for ((i = 0; i < 10; i++)); do
    eval "$cmd"
  done
}

function install_dllogger() {
  pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
  ldconfig
}

function install_torchrec_frameworks() {
  # FYI: Assuming machine has CUDA 12.4
  # If that is not true it could result in error

  # Install nvidia-cuda-toolkit
  # execute_with_retries "apt install -y nvidia-cuda-toolkit"

  run_multiple_times "pip uninstall -y torch fbgemm-gpu torchmetrics"

  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121
  pip install torchmetrics==1.0.3
  pip install torchrec --index-url https://download.pytorch.org/whl/cu121
}

function install_torchrec() {
  # pip uninstall -y torchrec
  dir_path="/projects"
  mkdir -p "$dir_path"
  pushd "$dir_path/"
  git clone --recursive https://github.com/pytorch/torchrec
  pushd "$dir_path/torchrec/"
  # pip install -r requirements.txt
  # python setup.py install develop
  popd
  popd
}

function install_llm_packages() {
  run_multiple_times "pip uninstall -y torchmetrics scipy"

  # --ignore-installed \
  # --no-dependencies \
  pip install \
    --no-cache-dir \
    huggingface-hub \
    accelerate \
    evaluate \
    fairscale \
    iopath \
    PyYAML \
    nvtabular \
    tensorrt \
    tokenizers \
    transformers==4.47.0 \
    safetensors \
    datasets \
    --ignore-installed llvmlite
  
  run_multiple_times "pip uninstall -y scipy matplotlib"

  pip install \
    --no-cache-dir \
    pytorch-lightning['extra'] \
    tensorboard \
    scipy==1.12.0 \
    matplotlib==3.5.3 \
    --ignore-installed llvmlite

  pip install \
    --no-cache-dir \
    git+https://github.com/Lightning-AI/torchmetrics.git@release/stable

  run_multiple_times "pip uninstall -y fastapi"

  pip install --no-cache-dir fastapi

}

function install_merlin_packages() {
  pip install \
    --no-cache-dir \
    merlin-core \
    merlin-models \
    merlin-systems \
    nvtabular \
    transformers4rec[nvtabular] \
    --ignore-installed llvmlite
}

function install_rapids_packages() {
  # https://docs.rapids.ai/install
  pip install --no-cache-dir \
    --no-dependencies \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.8.* dask-cudf-cu12==24.8.* cuml-cu12==24.8.* \
    cugraph-cu12==24.8.* cuspatial-cu12==24.8.* cuproj-cu12==24.8.* \
    cuxfilter-cu12==24.8.* cucim-cu12==24.8.* pylibraft-cu12==24.8.* \
    raft-dask-cu12==24.8.* cuvs-cu12==24.8.* \
    --ignore-installed llvmlite
}

function install_topicmodeling() {
  # Install `bertopic[flair,gensim,spacy,use]`
  pip install --no-cache-dir \
    --no-dependencies \
    bertopic \
    --ignore-installed llvmlite

  pip install --no-cache-dir \
    flair \
    gensim \
    spacy \
    --ignore-installed llvmlite
  
  # Set the path to the CUDA compiler
  export CUDACXX=/usr/local/cuda-12.4/bin/nvcc

  # Initialize LD_LIBRARY_PATH if it is not already set
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

  # Add CUDA and OpenMP library paths to LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/lib64/stubs:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

  # Install llama-cpp-python with specific CMake arguments
  CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir \
    llama-cpp-python

  dir_path="/projects"
  mkdir -p "$dir_path"
  pushd "$dir_path/"
  git clone --recursive https://github.com/TutteInstitute/datamapplot.git

  pip install --no-cache-dir \
    datamapplot/. \
    --ignore-installed llvmlite

  pip install --no-cache-dir \
    cupy-cuda12x \
    --ignore-installed llvmlite

  # Install spaCy model
  python -m spacy download en_core_web_sm

  # For GPU users
  pip install --no-cache-dir \
    --no-dependencies \
    tensorflow[and-cuda]
}

function reinstall_packages() {
  pip install --no-cache-dir \
    evidently==0.5.0
  pip install --no-cache-dir \
    dask-bigquery
  pip install --no-cache-dir \
    dask==2024.11.2

  run_multiple_times "pip uninstall -y numpy transformers matplotlib"

  pip install --no-cache-dir \
    numpy==1.26.4
  pip install --no-cache-dir \
    transformers==4.47.0
}

function install_core_packages() {
  pip install --no-cache-dir \
    google-api-core==2.19.0 \
    google-api-python-client==2.127.0 \
    google-auth==2.29.0 \
    google-auth-httplib2==0.2.0 \
    google-auth-oauthlib==1.0.0 \
    google-cloud-appengine-logging==1.4.3 \
    google-cloud-audit-log==0.2.5 \
    google-cloud-bigquery==3.21.0 \
    google-cloud-bigquery-storage==2.25.0 \
    google-cloud-core==2.4.1 \
    google-cloud-logging==3.10.0 \
    google-cloud-secret-manager==2.20.0 \
    google-cloud-storage==2.16.0 \
    gcsfs==2023.10.0 \
    joblib==1.4.2 \
    numpy==1.23.5 \
    pandas==1.5.3 \
    pandas-gbq==0.22.0 \
    psutil==5.9.8 \
    pyarrow==10.0.1 \
    pydata-google-auth==1.8.2 \
    scikit-learn==1.5.2 \
    tqdm==4.66.4 \
    xgboost==2.0.3 \
    joblibspark==0.5.2

  pip install --no-cache-dir \
    fire \
    --ignore-installed llvmlite
}


function main() {

  # 0. Check which python is used for Spark jobs
  if [[ -f /etc/profile.d/effective-python.sh ]]; then
    PROFILE_SCRIPT_PATH=/etc/profile.d/effective-python.sh
  elif [[ -f /etc/profile.d/conda.sh ]]; then
    PROFILE_SCRIPT_PATH=/etc/profile.d/conda.sh
  fi

  # 0.1 Ensure we have conda installed and available on the PATH
  if [[ -f "${PROFILE_SCRIPT_PATH}" ]]; then
    source "${PROFILE_SCRIPT_PATH}"
  fi

  # 0.2. Update /etc/profile.d/effective-python.sh if it exists
  if [[ -f /etc/profile.d/effective-python.sh ]]; then
    echo "Updating /etc/profile.d/effective-python.sh to use miniconda3..."
    sudo sed -i 's|/opt/conda/default/bin|/opt/conda/miniconda3/bin|g' /etc/profile.d/effective-python.sh
    echo "/etc/profile.d/effective-python.sh updated successfully!"
  fi

  # 0.3. Specify conda environment name (recommend leaving as root)
  if [[ ! -v CONDA_ENV_NAME ]]; then
    echo "No conda environment name specified, setting to 'root' env..."
    CONDA_ENV_NAME='root'
  # Force conda env name to be set to root for now, until a braver soul manages the complexity of environment activation
  # across the cluster.
  else
    echo "conda environment name is set to $CONDA_ENV_NAME"
    if [[ ! $CONDA_ENV_NAME == 'root' ]]; then
      echo "Custom conda environment names not supported at this time."
      echo "Force setting conda env to 'root'..."
    fi
    CONDA_ENV_NAME='root'
  fi

  # 0.4 Update PATH and conda...
  echo "Setting environment variables..."
  CONDA_BIN_PATH="/opt/conda/miniconda3/bin"
  export PATH="$CONDA_BIN_PATH:$PATH"
  echo "Updated PATH: $PATH"

  hash -r
  which conda
  conda config --set always_yes true --set changeps1 false

  # Useful printout for debugging any issues with conda
  conda info -a

  # 0.5 Update global profiles to add the miniconda location to PATH
  # based on: http://stackoverflow.com/questions/14637979/how-to-permanently-set-path-on-linux
  # and also: http://askubuntu.com/questions/391515/changing-etc-environment-did-not-affect-my-environemtn-variables
  # and this: http://askubuntu.com/questions/128413/setting-the-path-so-it-applies-to-all-users-including-root-sudo
  echo "Updating global profiles to export miniconda bin location to PATH..."
  echo "Adding path definition to profiles..."
  echo "# Environment varaibles set by Conda init action." | tee -a "${PROFILE_SCRIPT_PATH}" #/etc/*bashrc /etc/profile
  echo "export CONDA_BIN_PATH=$CONDA_BIN_PATH" | tee -a "${PROFILE_SCRIPT_PATH}"             #/etc/*bashrc /etc/profile
  echo 'export PATH=$CONDA_BIN_PATH:$PATH' | tee -a "${PROFILE_SCRIPT_PATH}"                 #/etc/*bashrc /etc/profile

  # 0.6 Update global profiles to add the miniconda location to PATH
  echo "Updating global profiles to export miniconda bin location to PATH and set PYTHONHASHSEED ..."
  # Fix issue with Python3 hash seed.
  # Issue here: https://issues.apache.org/jira/browse/SPARK-13330 (fixed in Spark 2.2.0 release)
  # Fix here: http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed/
  echo "Adding PYTHONHASHSEED=0 to profiles and spark-defaults.conf..."
  echo "export PYTHONHASHSEED=0" | tee -a "${PROFILE_SCRIPT_PATH}" #/etc/*bashrc  /usr/lib/spark/conf/spark-env.sh
  echo "spark.executorEnv.PYTHONHASHSEED=0" >>/etc/spark/conf/spark-defaults.conf

  ## 0.7 Ensure that Anaconda Python and PySpark play nice
  ### http://blog.cloudera.com/blog/2015/09/how-to-prepare-your-apache-hadoop-cluster-for-pyspark-jobs/
  echo "Ensure that Anaconda Python and PySpark play nice by all pointing to same Python distro..."
  echo "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python" | tee -a "${PROFILE_SCRIPT_PATH}" /etc/environment /usr/lib/spark/conf/spark-env.sh

  # 0.8 CloudSDK libraries are installed in system python
  echo 'export CLOUDSDK_PYTHON=/usr/bin/python' | tee -a "${PROFILE_SCRIPT_PATH}" #/etc/*bashrc /etc/profile

  # 1. Install required packages
  echo "Installing custom packages..."
  execute_with_retries "apt-get -y update"
  # install custom packages
  install_core_packages
  echo "Successfully installed custom packages."

  # Install torchrec packages
  if [[ "${INSTALL_TORCHREC}" == 'true' ]]; then
    # Install Frameworks
    install_torchrec_frameworks

    install_torchrec
  fi
  # torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script test_installation.py

  # install LLM packages
  if [[ "${INSTALL_LLM}" == 'true' ]]; then
    # Install LLM packages
    install_llm_packages
    echo "Successfully installed custom packages."
  fi

  # install Merlin packages
  if [[ "${INSTALL_MERLIN}" == 'true' ]]; then
    # Install Merlin packages
    install_merlin_packages
    echo "Successfully installed MERLIN packages."
  fi

  # install Rapids packages
  if [[ "${INSTALL_RAPIDS}" == 'true' ]]; then
    # Install Rapids packages
    install_rapids_packages
    echo "Successfully installed Rapids packages."
  fi

  # install NLP packages
  if [[ "${INSTALL_NLP}" == 'true' ]]; then
    # Install NLP packages
    install_topicmodeling
    echo "Successfully installed NLP packages."
  fi


  # 2. Append profiles with conda env source activate
  echo "Attempting to append ${PROFILE_SCRIPT_PATH} to activate conda env at login..."
  if [[ -f "${PROFILE_SCRIPT_PATH}" ]] && [[ ! $CONDA_ENV_NAME == 'root' ]]; then
    if grep -ir "source activate $CONDA_ENV_NAME" "${PROFILE_SCRIPT_PATH}"; then
      echo "conda env activation found in ${PROFILE_SCRIPT_PATH}, skipping..."
    else
      echo "Appending ${PROFILE_SCRIPT_PATH} to activate conda env $CONDA_ENV_NAME for shell..."
      sudo echo "source activate $CONDA_ENV_NAME" | tee -a "${PROFILE_SCRIPT_PATH}"
      echo "${PROFILE_SCRIPT_PATH} successfully appended!"
    fi
  elif [[ $CONDA_ENV_NAME == 'root' ]]; then
    echo "The conda env specified is 'root', the default environment, no need to activate, skipping..."
  else
    echo "No file detected at ${PROFILE_SCRIPT_PATH}..."
    echo "Are you sure you installed conda?"
    exit 1
  fi

  execute_with_retries "snap install nvtop"

  # Reinstall some packages to for exact version
  reinstall_packages

}

main
