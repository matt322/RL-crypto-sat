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
        
    def start(self, cnf_inst, decisions_per_callback=200000, timeout_secs=1000, args = []):
        self.inst = cnf_inst
        solverargs = [self.solver_path, cnf_inst[0], "-model", f"-decisions={decisions_per_callback}", "-verb=0"]
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

        reader = self.proc.stdout
        for line in reader:
            line = line.strip()
            if line.startswith("l "):
                clause = [int(x) for x in line.split()[1:]]
                self.fixed_clauses.append(clause)
                self.total_fixed += 1
            elif line.startswith("m "):
                if line == "m fixed done":
                    if self.verb > 0:
                        print(f"Read {self.total_fixed} fixed clauses")
                    break
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
        returns: (learnt_clauses, fixed_clauses, done, model, time since start)
        """
        if self.proc is None:
            raise RuntimeError("Process not started; call start() first")

        reader = self.proc.stdout
        writer = self.proc.stdin

        self.learnt_clauses = []
        self.total_learnts = 0
            
        reward = 0
        did_action = False

        if time.time() - self.start_time > self.timeout_secs:
            if self.verb > 0:
                print("Timeout reached, stopping solver")
            self.stop()
            return self.learnt_clauses, reward, True, None, time.time() - self.start_time

        for line in reader:
            line = line.strip()
            if line.startswith("l "):
                clause = [int(x) for x in line.split()[1:]]
                if self.reading_learnts:
                    self.learnt_clauses.append(clause)
                    self.total_learnts += 1
                else:
                    self.fixed_clauses.append(clause)
                    self.total_fixed += 1
            
            elif line.startswith("m "):
                match line:
                    case "m learnt done":
                        if self.verb > 0:
                            print(f"Read {self.total_learnts} learnt clauses")
                        if did_action:
                            return self.learnt_clauses, reward, False, None, time.time() - self.start_time                            
                    case "m learnt start":
                        self.total_learnts = 0
                        self.reading_learnts = True
                    case "m fixed done":
                        raise RuntimeError("Unexpected 'm fixed done' during step; should only appear during start()")
                    
                    case "m activity start":
                        if self.verb > 0:
                            print("Calculating activities...")
                        if activity_scores is None:
                            raise RuntimeError("Activity scores expected but not provided")
                        for i in activity_scores:
                            writer.write(i + "\n")
                        writer.write("m done\n")
                        writer.flush()
                        did_action = True

                    case _:
                        if line.startswith("m reward "):
                            reward = float(line.split(" ")[2])
                        if self.verb > 0:
                            print(f"Solver message: {line[2:]}")
            
            elif line.startswith("v "):
                if self.verb > 0:
                    print("solved")
                self.stop()
                return self.learnt_clauses, 100, True, line.split(" ")[1:], time.time() - self.start_time # solve reward
            
            elif line.startswith("s "): #only happens if unsat
                print(line)
                self.stop()
                return self.learnt_clauses, 0, True, None, time.time() - self.start_time

            else:
                print(line)
                    

    def zero_scores(self):
        return [f"{i} {0.0}" for i in range(self.inst[2])]

if __name__ == "__main__":
    inst = Instance(seed=41, rounds=21)
    solver = SolverController()
    cnf = inst.generate()
    #shutil.copy(cnf_path, "cnf/last_instance.cnf")
    runtime=0
    solver.start(cnf, 500000)
    while not solver._stop:
        learnts, reward, done, model, runtime = solver.step(solver.zero_scores())
    print(runtime)
        
    # for i in zip(model, true_model):
    #     if int(i[0]) != i[1]:
    #         #print("mismatch", i)
    

