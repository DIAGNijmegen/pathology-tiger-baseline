FROM nvidia/cuda:11.1-devel-ubuntu20.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-de -y \
    && apt-get clean \
    && :
    
# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl git \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.0-(Nightly)/ASAP-2.0-py38-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.0-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.0-py38-Ubuntu2004.deb \
    && :


# Libraries
RUN python -m pip install -U pip
RUN pip install git+https://github.com/DIAGNijmegen/pathology-hooknet 
RUN pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data 
RUN pip install tensorflow-gpu==2.3.0
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html 
RUN pip install numpy==1.20.0 
RUN pip install albumentations 
RUN pip install pycm 

# FOLDERS and PERMISSIONS
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /tempoutput /tempoutput/segoutput /tempoutput/detoutput /tempoutput/bulkoutput \
    /configs \
    && chown algorithm:algorithm /opt/algorithm /input /output /tempoutput /tempoutput/segoutput /tempoutput/detoutput \
    /tempoutput/bulkoutput

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# add scripts and models 
ADD --chown=algorithm:algorithm models /opt/algorithm/
ADD --chown=algorithm:algorithm configs /opt/algorithm/
COPY --chown=algorithm:algorithm write_annotations.py /opt/algorithm/
COPY --chown=algorithm:algorithm write_mask.py /opt/algorithm/
COPY --chown=algorithm:algorithm detectron_inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm concave_hull.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm association.py /opt/algorithm/
COPY --chown=algorithm:algorithm nms.py /opt/algorithm/


ENTRYPOINT python -u -m process $0 $@s

## ALGORITHM LABELS ##
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=bcsegdetrumc

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=('avx',)
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=30G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=15G
