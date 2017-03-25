FROM floydhub/dl-docker:cpu

ENV NODE_ENV docker

WORKDIR /opt

COPY . /opt

CMD python main.py
