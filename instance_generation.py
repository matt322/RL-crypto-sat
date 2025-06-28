import random
import os
import shutil
import subprocess
import hashlib


GLUCOSE_PATH = "glucose/simp/glucose"



def sha1_pad(binary_str: str) -> str:
    original_len = len(binary_str)
    assert original_len < 448
    binary_str += '1'
    
    while len(binary_str) != 448:
        binary_str += '0'
    
    original_len_bits = f"{original_len:064b}"
    binary_str += original_len_bits
    
    assert len(binary_str) == 512
    return binary_str

def random_input(rbytes=48):
    random_bits = ''.join(f'{byte:08b}' for byte in random.randbytes(rbytes))
    return sha1_pad(random_bits)


def generate(rounds=80):
    cnf_path = f"sha1_{rounds}round.cnf"
    copy_path = "copy.cnf"
    inputs = list(range(1, 512+1))
    input_str = random_input()
    for i in range(len(input_str)):
        inputs[i] *= 2 * int(input_str[i]) - 1
    outputs = list(range((rounds + 5) * 32 + 1, (rounds + 10) * 32 + 1))
    shutil.copy(cnf_path, copy_path)

    with open(copy_path, mode='a') as f:
        for i in inputs:
            f.write(f"{i} 0\n")
    
    soln = list(subprocess.run(f"{GLUCOSE_PATH} {copy_path} -model", text=True, shell=True, stdout=subprocess.PIPE).stdout.split(' '))
    soln = list(map(int, soln[soln.index('SATISFIABLE\nv')+1:-1]))
    os.remove(copy_path)
    output_str = ''.join('1' if soln[i-1] > 0 else '0' for i in outputs)
    return soln, input_str, output_str


def test():
    _, input, output = generate()
    hash = hashlib.sha1(int(input[:384], 2).to_bytes(48, 'big')).digest()
    print(input)
    print(output)
    print(''.join(f'{byte:08b}' for byte in hash))




test()
