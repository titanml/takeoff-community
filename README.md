<!-- <div align="center" id="top"> 
  <img src="./.github/app.gif" alt="Pantheon" />

  &#xa0; -->

  <!-- <a href="https://pantheon.netlify.app">Demo</a> -->
<!-- </div> -->

<h1 align="center">Fabulinus - FastAPI LLM Inference</h1>

<p align="center">
  <!-- <img alt="Github top language" src="https://img.shields.io/github/languages/top/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8"> -->

  <!-- <img alt="Github language count" src="https://img.shields.io/github/languages/count/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8"> -->

  <!-- <img alt="Repository size" src="https://img.shields.io/github/repo-size/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8"> -->

  <!-- <img alt="License" src="https://img.shields.io/github/license/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8"> -->

  <!-- <img alt="Github issues" src="https://img.shields.io/github/issues/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8" /> -->

  <!-- <img alt="Github forks" src="https://img.shields.io/github/forks/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8" /> -->

  <!-- <img alt="Github stars" src="https://img.shields.io/github/stars/{{YOUR_GITHUB_USERNAME}}/pantheon?color=56BEB8" /> -->
</p>

<!-- Status -->

<!-- <h4 align="center"> 
	🚧  Pantheon 🚀 Under construction...  🚧
</h4> 

<hr> -->

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
  <!-- <a href="#memo-license">License</a> &#xa0; | &#xa0; -->
  <a href="https://github.com/yc-wang00" target="_blank">Author</a>
</p>

<br>

## :dart: About ##

This repository hosts a FastAPI inference image, designed for demonstrating the deployment of large models, specifically those ranging from 7b to 40b parameters, using FastAPI. The setup optimizes a language model by using Ctranslate2. This resource is used by `iris takeoff` within the iris package.

## :sparkles: Features ##

:heavy_check_mark: Feature: Easy deployment and streaming response using FastAPI

## :rocket: Technologies ##

The following tools/package were used in this project:

- [Ctranslate2](https://github.com/OpenNMT/CTranslate2)
- [FastAPI](https://fastapi.tiangolo.com/)

## :white_check_mark: Requirements ##

Before starting :checkered_flag:, you need to have __docker__ and docker cudatoolkit (for gpu) installed. 

## :checkered_flag: Starting ##

```bash

# Access
$ cd fabulinus

# For dev, build the image first 
$ docker build -t <myimage> . 

# Spin up the container
$ docker run -it -p 8000:80 --gpus all -v /home/xxx/.iris_cache/xxx/:/code/models --entrypoint /bin/bash myimage:latest

# Notice this will run the CT2 convert and then spin up fastAPI server
$ sh run.sh 

# The server will initialize in the <http://localhost:8000>
```

You can then use `iris takeoff --infer` to test the inference 

## :hammer: Guide for Dev

### Updating the image in docker hub

1. build image first
```
docker build -t <myimage> .
```

2. tag the image
```
docker tag myimage:latest tytn/fabulinus:fastapi-launch
```

3. push the image to the hub
```
docker push tytn/fabulinus:fastapi-launch
```



<a href="#top">Back to top</a>
