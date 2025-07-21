from prefect import flow, task
import subprocess, pathlib

@task
def preprocess():
    subprocess.run(["python", "preprocess.py"], check=True)

@task
def train_baseline():
    subprocess.run(["python", "baseline.py"], check=True)

@task
def train_lora():
    subprocess.run(["python", "train_lora.py"], check=True)

@flow(name="nightly-retrain")
def nightly():
    preprocess()
    train_baseline()
    train_lora()

if __name__ == "__main__":
    nightly()
