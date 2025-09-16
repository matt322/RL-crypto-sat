import subprocess
import time
from instance_generation import Instance
import shutil

#init solver, will write learnts to stdout and take activity scores as input
#immediately yield non-learnt clauses (after simplification)
#l (clause)
#c (comment)
#m (message)


class SolverController:
    def __init__(self, solver_path="glucose_modified/simp/glucose", verb=0):
        self.solver_path = solver_path
        self.proc = None
        self._stop = False
        self.verb = verb
        
    def start(self, cnf_path, decisions_per_callback=200000, timeout_secs=1000):
        self.proc = subprocess.Popen(
            [self.solver_path, cnf_path, "-model", f"-decisions={decisions_per_callback}", "-verb=0", "-certified", "-certified-output=cnf/proof.txt"],
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


    def stop(self):
        self._stop = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def step(self, callback):
        """
        - read chunks from solver stdout (binary)
        - accumulate lines, parse only lines starting with b'l' or b'm'
        - when 'm done' seen, invoke callback(clauses)
        - write callback result (bytes/str) to solver stdin, flush
        - repeat until solver exits or stop() called
        returns: (learnt_clauses, fixed_clauses, done, model, time since start)
        """
        if self.proc is None:
            raise RuntimeError("Process not started; call start() first")

        reader = self.proc.stdout
        writer = self.proc.stdin

        self.learnt_clauses = []
        self.total_learnts = 0

        if time.time() - self.start_time > self.timeout_secs:
            if self.verb > 0:
                print("Timeout reached, stopping solver")
            self.stop()
            return self.learnt_clauses, self.fixed_clauses, True, None, time.time() - self.start_time

        for line in reader:
            line = line.strip()
            if line.startswith("m "):
                match line:
                    case "m learnt done":
                        if self.verb > 0:
                            print(f"Read {self.total_learnts} learnt clauses")
                    case "m learnt start":
                        self.total_learnts = 0
                        self.reading_learnts = True
                    case "m fixed done":
                        if self.verb > 0:
                            print(f"Read {self.total_fixed} fixed clauses")
                    case "m fixed start":
                        pass
                    case "m activity start":
                        if self.verb > 0:
                            print("Calculating activities...")
                        self._handle_batch(callback, writer)
                        return self.learnt_clauses, self.fixed_clauses, False, None, time.time() - self.start_time

                    case _:
                        if self.verb > 0:
                            print(f"Solver message: {line[2:]}")


            elif line.startswith("l "):
                clause = [int(x) for x in line.split()[1:]]
                if self.reading_learnts:
                    self.learnt_clauses.append(clause)
                    self.total_learnts += 1
                else:
                    self.fixed_clauses.append(clause)
                    self.total_fixed += 1
            
            elif line.startswith("v "):
                if self.verb > 0:
                    print("solved")
                self.stop()
                return self.learnt_clauses, self.fixed_clauses, True, line.split(" ")[1:], time.time() - self.start_time

            else:
                print(line)
                    
          
    def _handle_batch(self, callback, writer):
        """
        Call the callback and write the activity scores to solver stdin as index value pairs
        """
        resp = callback(self.fixed_clauses, self.learnt_clauses)
        for i in resp:
            writer.write(i + "\n")
        writer.write("m done\n")
        writer.flush()

if __name__ == "__main__":
    inst = Instance(seed=42, rounds=21)
    solver = SolverController()
    cnf_path, true_model, _, _ = inst.generate(p=0.01)
    shutil.copy(cnf_path, "cnf/last_instance.cnf")
    def callback(a, b):
        return [f"{i} {0.0}" for i in range(inst.nvars)]
    solver.start(cnf_path, 10000)
    while not solver._stop:
        learnts, fixed, done, model, runtime = solver.step(callback)
    for i in zip(model, true_model):
        if int(i[0]) != i[1]:
            print(i)
    
    

