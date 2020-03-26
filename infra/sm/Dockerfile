# DockerHub unaltered mirror of AWS Deep Learning Container
FROM 763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04

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


RUN apt-get update
RUN apt-get install -y --reinstall libglib2.0-0 libglib2.0-bin libglib2.0-data libsm6 libfontconfig1 libxrender1 libxext6

RUN git clone https://github.com/nikhilarunw/mask-rcnn-tensorflow /opt/ml/code/mask-rcnn-tensorflow
RUN chmod -R +w  /opt/ml/code/mask-rcnn-tensorflow
RUN pip install --ignore-installed -e  /opt/ml/code/mask-rcnn-tensorflow

# add custom nvidia coco tools
# need to be modified for pybind11 header files
RUN git clone https://github.com/NVIDIA/cocoapi && \
    cd cocoapi/PythonAPI && \
    make install

RUN pip install mpi4py
RUN pip install numba

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM run_mpi.py

