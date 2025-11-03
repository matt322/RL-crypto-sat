import random
import os
import shutil
import subprocess

class Instance:
    def __init__(self, rounds=21, solver_path="glucose_modified/simp/glucose"):
        self.solver_path = solver_path
        self.cnf_path = f"cnf/sha1_{rounds}round.cnf"
        self.rounds = rounds
        self.nvars, self.nclauses = self.get_varcount(self.cnf_path)
        self.instancefile = f"cnf/instance_{self.rounds}_rounds_{len(os.listdir("cnf"))}_{os.getpid()}.cnf"

    def __del__(self):
        try:
            os.remove(self.instancefile)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Could not remove {self.instancefile}: {e}")

    def get_varcount(self, path):
        with open(path, 'r') as f:
            first_line = f.readline().strip().split(' ')
            nvars = int(first_line[2])
            nclauses = int(first_line[3])
        f.close()
        return nvars, nclauses

    def sha1_pad(self, binary_str: str) -> str:
        original_len = len(binary_str)
        assert original_len < 448
        binary_str += '1'
        
        while len(binary_str) != 448:
            binary_str += '0'
        
        original_len_bits = f"{original_len:064b}"
        binary_str += original_len_bits
        
        assert len(binary_str) == 512
        return binary_str

    def random_input(self, rbytes=48):
        random_bits = ''.join(f'{byte:08b}' for byte in random.randbytes(rbytes))
        return self.sha1_pad(random_bits)


    def generate(self, free_outputs=0, guarantee_soln=False, simplify=False, seed=None):
        """
        Generate a new instance in the file associated to the object with variables revealed according to probability p

        Returns: (instancefile, solution, nvars, nclauses)
        """
        if seed:
            random.seed(seed)
        else:
            random.seed()
        assert 0 <= free_outputs <= 160, "160 output bits can be freed at most"
        inputs = list(range(1, 512+1))
        input_str = self.random_input()
        for i in range(len(input_str)):
            inputs[i] *= 2 * int(input_str[i]) - 1
        outputs = list(range((self.rounds + 5) * 32, (self.rounds + 10) * 32)) #variable names in cnf are 1 indexed, these are indices of the soln list
        free_outputs_list = random.sample(outputs, free_outputs)
        shutil.copy(self.cnf_path, self.instancefile)

        if guarantee_soln:
            with open(self.instancefile, mode='a') as f:
                for i in inputs:
                    f.write(f"{i} 0\n")
            soln = list(subprocess.run(f"{self.solver_path} {self.instancefile} -model", text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.split(' '))
            soln = list(map(int, soln[soln.index('SATISFIABLE\nv')+1:-1]))
            shutil.copy(self.cnf_path, self.instancefile)
            with open(self.instancefile, mode='a') as f:
                for i,v in enumerate(soln):
                    if i >= outputs[0] and i <= outputs[-1] and i not in free_outputs_list:
                        f.write(f"{v} 0\n")
        else:
            with open(self.instancefile, mode='a') as f:
                for i,v in enumerate(outputs):
                    if v not in free_outputs_list:
                        f.write(f"{random.choice((-1, 1)) * (v + 1)} 0\n")
            soln = None
            
        if simplify:
            subprocess.run(f"{self.solver_path} {self.instancefile} -dimacs={self.instancefile}", text=True, shell=True, stdout=subprocess.DEVNULL)
        nvars, nclauses = self.get_varcount(self.instancefile)
        return self.instancefile, soln, nvars, nclauses

if __name__ == "__main__":
    i = Instance(seed=42, rounds=21)
    i.generate(0.01)[1]
    shutil.copy(i.instancefile, "cnf/preimage_test.cnf")
    subprocess.run(f"{i.solver_path} cnf/preimage_test.cnf -certified-output=cnf/proof.txt", text=True, shell=True)
