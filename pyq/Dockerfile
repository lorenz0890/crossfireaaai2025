# Specify the parent image from which we build
FROM pytorch/pytorch

# Make the working directory
RUN mkdir /pyq
ADD . /pyq
WORKDIR /pyq

# Install addtinal pacakges
RUN pip install install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# Install PyQ
RUN pip install .
