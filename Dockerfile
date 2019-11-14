FROM tensorflow/tensorflow:1.2.1-gpu

WORKDIR /workdir
ADD requirements.txt /workdir
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT bash



