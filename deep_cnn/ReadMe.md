
# setup environment 
sudo pip install virtualenv      # This may already be installed
python3 -m venv .env             # Create a virtual environment (python3)
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies


.env/ virtualenv --system-site-packages  # Create a virtualenv environment
.env/ source ~/tensorflow/bin/activate   # Activate the TF environment

(tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
(tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
(tensorflow)$ pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU
(tensorflow)$ pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU

# get data ready
./get_datasets.sh

# launch jupyter notebook
source .env/bin/activate
.env/ jupyter notebook TensorFlow_CIFAR.ipynb
