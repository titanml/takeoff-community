<h1 align="center">TitanML | Takeoff Server</h1>




<p align="center">
  <img src="https://github.com/titanml/takeoff/assets/6034059/5b561d1a-7be3-4258-bd4d-bb670fdb2c1e" alt="Image from TitanML">
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

## 🚊 Usage

To use the inference server, use the `iris` launcher. To install, run 
```
pip install titan-iris
```
Then, to launch an inference server with a model, run 
```
iris takeoff --model tiiuae/falcon-7b-instruct --device cpu --port 8000
```
You'll be prompted to login. To run with GPU access, add `--device cuda` instead. 

To experiment with the resulting server, navigate to http://localhost:8000/demos/playground, or http://localhost:8000/demos/chat. To see docs on how to query the model, navigate to http://localhost:8000/docs

## :checkered_flag: Developing ##

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
