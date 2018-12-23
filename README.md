# Rasa Stack starter-pack

Looked through the [Rasa NLU](http://rasa.com/docs/nlu/) and [Rasa Core](http://rasa.com/docs/core/) documentation and ready to build your first intelligent assistant? We have some resources to help you get started! This repository contains the foundations of your first custom assistant.  

This starter-pack comes with a small amount of training data which lets you build a simple assistant. **You can find more training data here in the [forum](https://forum.rasa.com/t/grab-the-nlu-training-dataset-and-starter-packs/903) and use it to teach your assistant new skills and make it more engaging.**

We would recommend downloading this before getting started, although the tutorial will also work with just the data in this repo. 

The initial version of this starter-pack lets you build a simple assistant capable of cheering you up with Chuck Norris jokes.


<p align="center">
  <img src="./rasa-stack-mockup.gif">
</p>


Clone this repo to get started:

```
git clone https://github.com/RasaHQ/starter-pack-rasa-stack.git
```

After you clone the repository, a directory called starter-pack-rasa-stack will be downloaded to your local machine. It contains all the files of this repo and you should refer to this directory as your 'project directory'.


## Setup and installation

필요 Package 설치 
- rasa_nlu, rasa_core, konlpy

```
pip install -r requirements.txt
```

Mecab 설치

- package가 설치된 폴더로 이동 하여 mecab 설치
   - ex) lib/python3/site-packages
   - 설치에 다음 경로 활용 -> https://bitbucket.org/eunjeon/

```
>>> git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git 
>>> cd mecab-python-0.996
>>> python setup.py build
>>> python setup.py install
```

