<h1 align="center">TitanML | Takeoff Server</h1>




<p align="center">
  <img src="https://github.com/titanml/takeoff/assets/6034059/044f1a7e-7deb-46d9-9618-c8327124d397" alt="Image from TitanML">
</p>

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-community-features">Features</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; &#xa0;
</p>

## :dart: About ##

This is the repository for the community edition of the TitanML Takeoff server. This is a server designed for optimized inference of large language models. 

For more information, see the [docs](https://docs.titanml.co/docs/titan-takeoff/getting-started).

## :sparkles: Community Features ##

:heavy_check_mark: Easy deployment and streaming response

:heavy_check_mark: Optimized int8 quantization

:heavy_check_mark: Chat and playground-like interface

:heavy_check_mark: Support for encoder-decoder (T5 family) and decoder models

For the pro edition, including multi-gpu inference, int4 quantization, and more. [contact us](mailto:hello@titanml.co)

## :white_check_mark: Requirements ##

Before getting started :checkered_flag:, you need to have __docker__ and docker cudatoolkit (for gpu) installed. 

## :checkered_flag: Working with the container ##

```bash

# Access
$ cd takeoff

# For dev, build the image first 
$ docker build -t <myimage> . 

# Spin up the container
$ docker run -it -p 8000:80 --gpus all -v $HOME/.iris_cache/:/code/models/ -v  --entrypoint /bin/bash myimage:latest

# set the models and device
export MODEL_NAME=t5-small
export DEVICE=cuda # or cpu

# This will run the CT2 convert and then spin up the fastAPI server
$ sh run.sh 

# The server will initialize in the <http://localhost:8000>
```

You can then use `iris takeoff --infer` to test the inference 

For more details as to how to use the server, check out the [docs](https://docs.titanml.co)

<a href="#top">Back to top</a>
