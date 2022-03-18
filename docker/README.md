# Docker for dlup_lightning_mil
A Dockerfile is provided for dlup_lightning_mil which provides you with all the required dependencies.

## Pull the docker image from dockerhub
```
docker pull yonischirris/deepsmile:mil-latest
```

## Build the image

Clone the repository
```
git clone git@github.com:NKI-AI/dlup-lightning-mil.git
```

Move into the project's root directory
```
cd dlup-lightning-mil
```

Set up the submodules
```
git submodule update --init --recursive
```

Build the container
```
docker build -t dlup_lightning_mil:latest . -f docker/Dockerfile
```

or, for MacOS with M1 chip:
```
docker build --platform linux/amd64 -t dlup_lightning_mil:latest . -f docker/Dockerfile
```

Running the container can for instance be done with:
```
docker run -it --ipc=host --rm -v /data:/data dlup_lightning_mil:latest /bin/bash
```

To use the container to train on an HPC cluster, save it as a `.tar` and export it to a singularity file

```
docker save container_id -o container_name_YYYYMMDD.tar
singularity build container_name_YYYYMMDD.sif docker-archive://container_name_YYYYMMDD.tar
```

If you have changed code in this repo or any of its submodules, bind the repo to `/hissl-lightning-cli` to use
the edited version inside the docker.

To use a singularity image to reproduce the results from DeepSMILE, make sure it is saved as `~/hissl/mil_deepsmile.sif`.
