# Use the official Nvidia CUDA image as devel
ARG CUDA
FROM nvidia/cuda:${CUDA}-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    software-properties-common

# Add the LLVM APT repository
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN add-apt-repository "deb http://apt.llvm.org/focal/ llvm-toolchain-focal main"
    
# Install the latest version of Clang
RUN apt-get update && apt-get install -y clang llvm

# Set the working directory
WORKDIR /src

# Copy the current directory contents into the container
COPY . /src

# Set the default command to run when starting a container
CMD ["bash"]
