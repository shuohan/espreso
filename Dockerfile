FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN pip install scipy nibabel matplotlib ipython
RUN apt-get update && apt-get install -y git graphviz
RUN pip install git+https://github.com/shuohan/pytorchviz.git
RUN pip install git+https://github.com/shuohan/singleton-config.git
RUN pip install graphviz
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
