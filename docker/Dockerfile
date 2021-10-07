FROM ubuntu:20.04
COPY requirements.txt /exp/requirements.txt
COPY ./mnist/modelling.py /exp/modelling.py
RUN apt-get update && apt-get install -y python3-pip && apt-get clean
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/models
WORKDIR /exp
CMD ["python3", "./mnist/modelling.py"]
