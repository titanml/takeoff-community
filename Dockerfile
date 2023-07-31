# 
FROM ghcr.io/opennmt/ctranslate2:3.17.1-ubuntu20.04-cuda11.2

# 
WORKDIR /code

RUN mkdir /code/models

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# 
COPY ./app /code/app
COPY ./run.sh /code/run.sh
#
ENTRYPOINT ["sh", "run.sh"]


