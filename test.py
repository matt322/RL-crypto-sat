import os
import shutil
import subprocess

GLUCOSE_PATH = "glucose_modified/simp/glucose"




def run_glucose(cnf_path):
    return subprocess.run(f"{GLUCOSE_PATH} {cnf_path} -model", text=True, shell=True, stdout=subprocess.PIPE).stdout.split(' ')

run_glucose("examplepreimage.cnf")