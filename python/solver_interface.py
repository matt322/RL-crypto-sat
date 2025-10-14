import subprocess
import time
from instance_generation import Instance
from multiprocessing import shared_memory, resource_tracker
import numpy as np
import gc

#init solver, will write learnts to stdout and take activity scores as input
#immediately yield non-learnt clauses (after simplification)
#l (clause)
#c (comment)
#m (message)


class SolverController:
    def __init__(self, solver_path="glucose_modified/simp/glucose"):
        self.solver_path = solver_path
        self.proc = None
        self._stop = False
        
    def start(self, cnf_inst, decisions_per_callback=200000, timeout_secs=2**31, args = [], verb=0):
        self.verb=verb
        self._stop = False
        self.inst = cnf_inst
        solverargs = [self.solver_path, cnf_inst[0], "-model", f"-decisions={decisions_per_callback}", f"-verb={0 if self.verb < 2 else 1}"]
        if timeout_secs < 2**31:
            solverargs.append(f"-cpu-lim={timeout_secs}")
        solverargs.extend(args)
        self.proc = subprocess.Popen(
            solverargs,
            stdin=subprocess.PIPE,      
            stdout=subprocess.PIPE,     
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Failed to open pipes to subprocess")
        self.fixed_clauses = []
        self.learnt_clauses = []
        self.reading_learnts = False
        self.total_learnts = 0
        self.total_fixed = 0
        self.start_time = time.time()
        self.timeout_secs = timeout_secs
        self.log = []

        self.reader = self.proc.stdout
        self.writer = self.proc.stdin

        for line in self.reader:
            line = line.strip()
            if self.verb > 3:
                print(line)
            if line.startswith("l "):
                clause = [int(x) for x in line.split()[1:]]
                self.fixed_clauses.append(clause)
                self.total_fixed += 1
            elif line.startswith("m "):
                if line == "m fixed done":
                    if self.verb > 0:
                        print(f"Read {self.total_fixed} fixed clauses")
                    break
                if line.startswith("m shared_mem name"):
                    self.shm_name = line.split(" ")[3][1:]
                if line.startswith("m csr written"):
                    if self.verb > 1:
                        print(f"Reading CSR from shared memory: {self.shm_name}")
                    csr = self.read_from_shared_mem(self.shm_name)
                    return csr, 0, False, None, time.time() - self.start_time
        return self.fixed_clauses
                

        

    def stop(self):
        self._stop = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def is_finished(self):
        return self._stop

    def step(self, activity_scores=None):
        """
        - read chunks from solver stdout (binary)
        - accumulate lines, parse only lines starting with b'l' or b'm'
        - when 'm done' seen, invoke callback(clauses)
        - write callback result (bytes/str) to solver stdin, flush
        - repeat until solver exits or stop() called
        returns: (learnt_clauses, reward, done, model, time since start)
        """
        if self.proc is None:
            raise RuntimeError("Process not started; call start() first")

        self.learnt_clauses = []
        self.total_learnts = 0
            
        reward = 0
        did_action = False

        if time.time() - self.start_time > self.timeout_secs:
            if self.verb > 0:
                print("Timeout reached, stopping solver")
            self.stop()
            return self.learnt_clauses, reward, True, None, time.time() - self.start_time

        for line in self.reader:
            line = line.strip()
            
            if line.startswith("l "):
                if self.verb > 3:
                    print(line)
                clause = [int(x) for x in line.split()[1:]]
                if self.reading_learnts:
                    self.learnt_clauses.append(clause)
                    self.total_learnts += 1
                else:
                    self.fixed_clauses.append(clause)
                    self.total_fixed += 1

            elif line.startswith("m "):
                if self.verb > 2:
                    print(line)
                match line:
                    case "m learnt done":
                        if self.verb > 1:
                            print(f"Read {self.total_learnts} learnt clauses")
                        if did_action:
                            return self.learnt_clauses, reward, False, None, time.time() - self.start_time                            
                    case "m learnt start":
                        self.total_learnts = 0
                        self.reading_learnts = True
                    case "m fixed done":
                        raise RuntimeError("Unexpected 'm fixed done' during step; should only appear during start()")
                    
                    case "m activity start":
                        if self.verb > 2:
                            print("Calculating activities...")
                        if activity_scores is None:
                            raise RuntimeError("Activity scores expected but not provided")
                        for i in activity_scores:
                            self.writer.write(i + "\n")
                        self.writer.write("m done\n")
                        self.writer.flush()
                        did_action = True

                    case _:
                        if line.startswith("m shared_mem name"):
                            self.shm_name = line.split(" ")[3][1:]
                        if line.startswith("m csr written"):
                            csr = self.read_from_shared_mem(self.shm_name)
                            if self.verb > 1:
                                print(f"Read {csr["nclauses"]} clauses from shared memory")
                            return csr, 0, False, None, time.time() - self.start_time
                        if line.startswith("m reward "):
                            reward = float(line.split(" ")[2])
                        if self.verb > 2:
                            print(f"Solver message: {line[2:]}")
            
            elif line.startswith("v "):
                if self.verb > 1:
                    print(line)
                self.stop()
                return self.learnt_clauses, 100, True, line.split(" ")[1:], time.time() - self.start_time # solve reward

            else:
                if self.verb > 0:
                    print(line)
        return self.learnt_clauses, 0, True, None, time.time() - self.start_time #process exited


    def read_from_shared_mem(self, shm_name):
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = shm.buf

        header = np.frombuffer(buf, dtype=np.int32, count=3, offset=0)
        n_clauses = int(header[0])
        n_lits = int(header[1])
        nnz = int(header[2])
        crow_bytes = (n_clauses + 1) * 4

        off_crow = 3 * 4
        off_col = off_crow + crow_bytes
        crow_arr = np.frombuffer(buf, dtype=np.int32, count=n_clauses + 1, offset=off_crow).copy()
        col_arr  = np.frombuffer(buf, dtype=np.int32, count=nnz, offset=off_col).copy()

        del buf
        del header

        self.writer.write("m csr ack\n")
        self.writer.flush()
        resource_tracker.unregister(shm._name, "shared_memory")
        shm.close()

        return {"crow_indices": crow_arr,
                "col_indices": col_arr,
                "values": np.ones(nnz, dtype=np.int32),
                "nlits": n_lits,
                "nclauses": n_clauses,
                "nnz": nnz
            }


    def zero_scores(self):
        return [f"{i} {0.0}" for i in range(self.inst[2])]

def verify_cnf():
    inst = ["cnf/test.cnf", [], 40, 180]
    solver = SolverController()
    print(solver.start(inst, 100, verb=2))
    while not solver.is_finished():
        print(solver.step(solver.zero_scores()))


if __name__ == "__main__":
    verify_cnf()
    exit()

    inst = Instance(seed=41, rounds=21)
    solver = SolverController()
    cnf = inst.generate()
    #model = cnf[1]
    solver.start(cnf, 50000, timeout_secs=40, verb=2)
    for i in range(10):
        solver.step(solver.zero_scores())

   
    

