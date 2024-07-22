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

- `configs/default.yaml` : Gaia parameter
- `configs/ds_config.json` : Deepspeed setting 
- `scripts/ds_train.sh` : Job script for ABCI

> [!CAUTION] 
> You must edit below 3 points.

- Row 88 in train.py: 
    - 88: `YOUR_PROJECT` and `YOUR_ACCOUNT_NAME`

- Row 2 and 17 in configs/default.yaml: 
    - 2: `WANDB_RUN_NAME`
    - 17: `PATH_TO_YOUR_GROUP_DIRECTORY`

- Row 6 and 17 in scripts/ds_train.sh
    - 6: `WANDB_RUN_NAME`
    - 17: `PATH_TO_GAIA_DIRECTORY`


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