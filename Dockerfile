FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get install -y git
RUN pip install scipy==1.5.2 \
                nibabel==3.1.1 \
                matplotlib==3.3.2 \
                pandas==1.1.3 \
                sphinx==3.2.1 \
                sphinx-rtd-theme==0.5.0 \
                git+https://github.com/shuohan/ssp@0.1.0 \
                git+https://github.com/shuohan/pytorch-trainer@0.1.1 \
                git+https://github.com/shuohan/sssrlib@0.1.1 \
                git+https://github.com/shuohan/improc3d@0.5.0 \
                git+https://github.com/shuohan/singleton-config.git@0.2.0
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
