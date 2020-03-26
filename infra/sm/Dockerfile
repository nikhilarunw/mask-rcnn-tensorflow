# DockerHub unaltered mirror of AWS Deep Learning Container
FROM 763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-training:2.1.0-gpu-py36-cu101-ubuntu18.04
RUN pip install keras h5py
RUN pip install Cython
RUN pip install ujson opencv-python matplotlib
RUN pip install --ignore-installed numpy==1.16.2
RUN pip install pybind11
RUN pip install mpi4py
RUN pip install numba

# For Sagemaker
RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY run_mpi.py /opt/ml/code/run_mpi.py
COPY run.sh /opt/ml/code/run.sh
RUN chmod +x /opt/ml/code/run.sh

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow /opt/ml/code/mask-rcnn-tensorflow
RUN chmod -R +w  /opt/ml/code/mask-rcnn-tensorflow
RUN pip install --ignore-installed -e  /opt/ml/code/mask-rcnn-tensorflow

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM run_mpi.py
