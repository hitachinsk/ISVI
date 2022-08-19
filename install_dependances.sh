#!/bin/bash
cd forward_warp/cuda
python setup.py install | grep "error"
cd ../../
python setup.py install
cd ..
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
tar xvf ffmpeg-git-amd64-static.tar.xz && \
rm ffmpeg-git-amd64-static.tar.xz && \
mv ffmpeg*/ffmpeg ffmpeg*/ffprobe /opt/conda/bin/ && \