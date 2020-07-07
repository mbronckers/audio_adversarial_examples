- Follow Carlini Instructions (below) and check below for any dependency issues I faced when installing on Nebula server

- in case you are pulling my github repo (and the included submodule DeepSpeech), I believe you need to run
    `git clone -recursive [this repo]`
    or 
    `git submodule init && git submodule update` after cloning without recursive

### Resolving dependency hell

- Use virtualenv for convenience
- cd into DeepSpeech for 1a 2)
- tensorflow-gpu in case of NVDIA GPU. otherwise just tensorflow==1.14

- decoder module missing => download most recent ds_ctcdecoder via https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.0-alpha.2 and just `pip install [file]`
    - `No module named 'ds_ctcdecoder'` error =>
    `pip install https://github.com/mozilla/DeepSpeech/releases/download/v0.9.0-alpha.2/ds_ctcdecoder-0.9.0a2-cp37-cp37m-manylinux1_x86_64.whl`

- install git-lfs:
    ` wget https://github.com/github/git-lfs/releases/download/v1.2.0/git-lfs-linux-amd64-1.2.0.tar.gz`
    `tar -xzf git-lfs-linux-amd64-1.2.0.tar.gz`

NOTE:

- if you put the DeepSpeech model inside the DeepSpeech folder (like me), change the path of `--restore_path` to `DeepSpeech/deepspeech-0.4.1(...)`

### Carlini Instructions

Instructions for basic use:

1a. Install the dependencies

$ pip3 install tensorflow-gpu==1.14 progressbar numpy scipy pandas python_speech_features tables attrdict pyxdg

$ pip3 install $(python3 util/taskcluster.py --decoder)

Download and install
https://git-lfs.github.com/

1b. Make sure you have installed git lfs. Otherwise later steps will mysteriously fail.

2. Clone the Mozilla DeepSpeech repository into a folder called DeepSpeech:

git clone https://github.com/mozilla/DeepSpeech.git

2b. Checkout the correct version of the code:

(cd DeepSpeech; git checkout tags/v0.4.1)

2c. If you get an error with tflite_convert, comment out DeepSpeech.py Line 21
`# from tensorflow.contrib.lite.python import tflite_convert`

3. Download the DeepSpeech model

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
tar -xzf deepspeech-0.4.1-checkpoint.tar.gz

4. Verify that you have a file deepspeech-0.4.1-checkpoint/model.v0.4.1.data-00000-of-00001
Its MD5 sum should be
ca825ad95066b10f5e080db8cb24b165

5. Check that you can classify normal images correctly

python3 attack.py --in sample-000000.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1

6. Generate adversarial examples

python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1

8. Verify the attack succeeded

python3 attack.py --in adv.wav --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1