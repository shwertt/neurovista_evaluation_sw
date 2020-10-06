FROM tensorflow/tensorflow:2.0.1-gpu-py3
WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt