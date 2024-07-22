# Jeong Solution of nanoGPT-hackathon

Original repository: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

# Quik Start
## Set Up
Creation of a virtual environment: work
```
cd PATH_TO_WORKING_DIR 
```
```
module load python/3.11/3.11.9 cuda/12.1/12.1.1 cudnn/8.9/8.9.7
```
```
python3 -m venv work
```
Activating a virtual environment: work
```
source work/bin/activate
```
Installing dependencies.
```
pip3 install -r requirements.txt
```
Deactivating a virtual environment
```
deactivate
```
## Multinode Setting
> [!NOTE]
> You can set training parameter in below files.

> [!CAUTION] 
> You must edit below files.

## Multinode Training
```
cd PATH_TO_WORKING_DIR
```
```
qsub -g gcb50389 scripts/train.sh
```
then you can see training status 
```
cat WANDB_RUN_NAME.oxxxxx