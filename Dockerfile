FROM nvidia/cuda:11.1-devel-ubuntu18.04

# FOLDERS and PERMISSIONS
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /tempoutput /tempoutput/segoutput /tempoutput/detoutput /tempoutput/bulkoutput \
    /configs \
    && chown algorithm:algorithm /opt/algorithm /input /output /tempoutput /tempoutput/segoutput /tempoutput/detoutput \
    /tempoutput/bulkoutput

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Libraries
RUN python3 -m pip install --user -U pip
RUN pip install git+https://github.com/DIAGNijmegen/pathology-hooknet --user
RUN pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data --user
RUN pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html --user
RUN pip install numpy==1.20.0 --user
RUN pip install albumentations --user
RUN pip install pycm --user

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


ENTRYPOINT python3 -u -m process $0 $@

## ALGORITHM LABELS ##
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=bcsegdetrumc

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=('avx',)
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=30G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=15G
