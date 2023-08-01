<h1 align="center">Fabulinus - FastAPI LLM Inference</h1>

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
</p>

<br>

## :dart: About ##

This is the repository for the community edition of the TitanML Takeoff server. This is a server designed for optimized inference of large language models. For single GPU and CPU deployment, we use [CTranslate2](https://opennmt.net/CTranslate2/index.html), an inference engine for transformer models. 

## :sparkles: Features ##

:heavy_check_mark: Easy deployment and streaming response

:heavy_check_mark: Optimized int8 quantization

:heavy_check_mark: Chat and playground-like interface

:heavy_check_mark: Support for encoder-decoder (T5 family) and decoder models

## :white_check_mark: Requirements ##

Before starting :checkered_flag:, you need to have __docker__ and docker cudatoolkit (for gpu) installed. 

## :checkered_flag: Working with the container ##

```bash

# Access
$ cd takeoff

# For dev, build the image first 
$ docker build -t <myimage> . 

# Spin up the container
$ docker run -it -p 8000:80 --gpus all -v /home/xxx/.iris_cache/xxx/:/code/models --entrypoint /bin/bash myimage:latest

# set the models and device
export MODEL_NAME=t5-small
export DEVICE=cuda # or cpu

# This will run the CT2 convert and then spin up the fastAPI server
$ sh run.sh 

# The server will initialize in the <http://localhost:8000>
```

You can then use `iris takeoff --infer` to test the inference 

For more details as to how to use the server, check out the 

<a href="#top">Back to top</a>
