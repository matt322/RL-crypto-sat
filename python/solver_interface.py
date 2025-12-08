import subprocess
import time
from instance_generation import Instance
from multiprocessing import shared_memory, resource_tracker
import numpy as np
import yappi

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
        
    def start(self, cnf_inst, decisions_per_callback=0, simplify_clauses=False, get_csr=True, timeout_secs=2**31, reward_pow = 2, args = [], verb=0):
        self.verb=verb
        self.stop()
        self._stop = False
        self.inst = cnf_inst
        solverargs = [self.solver_path, cnf_inst[0], "-model", f"-decisions={decisions_per_callback}", f"-verb={0 if self.verb < 2 else 1}", f"-reward_pow={reward_pow}"]
        if simplify_clauses:
            solverargs.append("-simplify_clauses")
        if timeout_secs < 2**31:
            solverargs.append(f"-cpu-lim={timeout_secs}")
        solverargs.extend(args)
        
        self.proc = subprocess.Popen(
            solverargs,
            stdin=subprocess.PIPE,      
            stdout=subprocess.PIPE,     
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Failed to open pipes to subprocess")
       
        self.start_time = time.time()
        self.timeout_secs = timeout_secs
        self.log = []

        self.reader = self.proc.stdout
        self.writer = self.proc.stdin
        self.err = self.proc.stderr

        self.debug_lines = []
        for line in self.reader:
            line = line.strip()
            self.debug_lines.append(line)
            if self.verb > 3:
                print(line)
            if line.startswith("c"):
                continue
            elif line.startswith("m "):
                if line == "m csr ready":
                    self.writer.write(f"m csr {"yes" if get_csr else "no"}\n")
                    self.writer.flush()

                elif line.startswith("m shared_mem name"):
                    assert get_csr
                    self.shm_name = line.split(" ")[3][1:]

                elif line.startswith("m csr written"):
                    assert get_csr
                    csr = self.read_from_shared_mem(self.shm_name)
                    if self.verb > 1:
                        print(f"Read {csr["n_clauses"]} clauses from {self.shm_name}")
                    return csr, 0, False, False, None, time.time() - self.start_time

            elif line.startswith("s ") or line.startswith("v "):
                if self.verb > 1:
                    print(line)
                self.stop()
                return None, 1, True, False, line.split(" ")[1:], time.time() - self.start_time 
            
        print(f"Solver exited unexpectedly in start(): poll returned {self.proc.poll()}")
        for errline in self.err:
            print(f"ERR: {errline.strip()}")
        for debugline in self.debug_lines:
            print(f"DBG: {debugline}")
        self.stop()
        return None, 0, True, False, None, time.time() - self.start_time
        

    def stop(self):
        self._stop = True
        if self.proc:
            try: 
                #self.writer.write("m csr ack\n")
                #self.writer.flush()
                self.proc.terminate()
            except Exception:
                pass

    def is_finished(self):
        return self._stop

    def step(self, activity_scores, go_ahead=None, get_csr=True, heuristic_type="refocus", static_score_decisions=0):
        """
        sends queries to solver after decision interval has been reached. activity score action -> go forward -> solver runs -> read csr and reward
        returns: (learnt obs object, reward, done, timed out, model, time since start)
        """

        if self.proc is None or self.is_finished():
            raise RuntimeError("Process not started; call start() first")

        self.total_learnts = 0
        csr = None
        reward = None
        start_step_time = time.time()
        
        if time.time() - self.start_time > self.timeout_secs:
            if self.verb > 0:
                print("Timeout reached, stopping solver")
            self.stop()
            return None, reward, True, True, None, time.time() - start_step_time

        for line in self.reader:
            line = line.strip()
            self.debug_lines.append(line)
            if line.startswith("m "):
                if self.verb > 2:
                    print(line)
                if line == "m activity start": 
                    if heuristic_type == "refocus":
                        self.writer.write("m scores vsids\n") #specify array to write to
                        for i in activity_scores:
                            self.writer.write(i + "\n")
                        self.writer.write("m activity done\n")
                        self.writer.flush()
                        
                    elif heuristic_type == "hybrid":
                        self.writer.write("m scores static\n")
                        for i in activity_scores:
                            self.writer.write(i + "\n")
                        self.writer.write(f"m static_decisions {static_score_decisions}\n")
                        self.writer.write("m activity done\n")
                        self.writer.flush()

                    else:
                        raise ValueError("Heuristic type unknown")
                    
                    if go_ahead is not None:
                            assert isinstance(go_ahead, int) and go_ahead > 0
                            self.writer.write(f"m forward {go_ahead}\n")
                            self.writer.flush()
                    else:
                            self.writer.write(f"m continue\n")
                            self.writer.flush()
                
                elif line == "m step done":
                    assert (csr is not None or not get_csr) and reward is not None, "step done recieved before obs or reward"
                    return csr, reward, False, False, None, time.time() - start_step_time

                elif line == "m csr ready":
                    self.writer.write(f"m csr {"yes" if get_csr else "no"}\n")
                    self.writer.flush()

                elif line.startswith("m shared_mem name"):
                    assert get_csr
                    self.shm_name = line.split(" ")[3][1:]

                elif line.startswith("m csr written"):
                    assert get_csr
                    csr = self.read_from_shared_mem(self.shm_name)
                    if self.verb > 1:
                        print(f"Read {csr["n_clauses"]} clauses from {self.shm_name}")
    
                elif line.startswith("m reward "):
                    reward = float(line.split(" ")[2])
                else:
                    print(line)
                    
            
            elif line.startswith("v "): #solver found model
                if self.verb > 1:
                    print(line)
                self.stop()
                return None, 1 if reward is None else reward, True, False, line.split(" ")[1:], time.time() - start_step_time # solve reward
            
            elif line.startswith("s "): #timed out
                if self.verb > 1:
                    print(line)
                self.stop()
                return None, 0 if reward is None else reward, True, True, line.split(" ")[1:], time.time() - start_step_time # solve reward
            
            else:
                if self.verb > 0:
                    print(line)

        print(f"Solver exited unexpectedly in step: poll returned {self.proc.poll()}")
        print(f"Last line: {self.debug_lines[-1]}")
        if self.verb > 1:
            for errline in self.err:
                print(f"ERR: {errline.strip()}")
            for debugline in self.debug_lines:
                print(f"DBG: {debugline}")
        self.stop()
        return None, 0 if reward is None else reward, True, True, None, time.time() - start_step_time #process exited


    def read_from_shared_mem(self, shm_name):
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = shm.buf

        header = np.frombuffer(buf, dtype=np.int32, count=3, offset=0).copy()
        n_clauses = int(header[0]) - 1
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
                "n_clauses": n_clauses,
                "nnz": nnz
            }


    def zero_scores(self):
        return [f"{i} {1e-8}" for i in range(self.inst[2])]
    
    def random_scores(self):
        return [f"{i} {np.random.random()}" for i in range(self.inst[2])] 
    
    def nothing_scores(self):
        return []

def verify_cnf():
    inst = ["cnf/test.cnf", [], 15, 67]
    solver = SolverController()
    print(solver.start(inst, 1, verb=2, simplify_clauses=False))
    while not solver.is_finished():
        solver.step(solver.zero_scores())

def verify_activity_scores():
    inst = Instance()
    cnf = inst.generate(seed = 41, guarantee_soln=True)
    scores = [f"{i} {1 if v < 0 else 0}" for i,v in enumerate(cnf[1])]
    solver = SolverController()
    solver.start(cnf, 10000, verb=0)
    assert solver.step(scores)[2] == True

def verify_hybrid():
    inst = Instance()
    cnf = inst.generate(seed = 41, guarantee_soln=True)
    scores = [f"{i} {1 if v < 0 else 0}" for i,v in enumerate(cnf[1])]
    solver = SolverController()
    solver.start(cnf, 10000, verb=0)
    assert solver.step(scores, heuristic_type="hybrid", static_score_decisions=4000)[2] == True


if __name__ == "__main__":
    verify_activity_scores()
    verify_hybrid()

    inst = Instance(rounds=21)
    solver = SolverController()
    cnf = inst.generate(seed=41)
    yappi.clear_stats()
    yappi.set_clock_type("wall")
    yappi.start()
    solver.start(cnf, 500000, timeout_secs=40, verb=0, simplify_clauses=True, reward_pow=0)
    yappi.stop()
    yappi.get_func_stats().print_all()
    timetot = 0
    while not solver.is_finished():
        _, r, _, _, _, t = solver.step([])
        timetot += t
        print(f"Step reward: {r}")
    print(f"Total time: {timetot}")

   
    

