import subprocess
import threading
import os
from instance_generation import Instance

#init solver, will write learnts to stdout and take activity scores as input
#immediately yield non-learnt clauses (after simplification)
#l (clause)
#c (comment)
#m (message)


class SolverController:
    def __init__(self, bufsize: int = 65536, solver_path="glucose_modified/simp/glucose"):
        self.solver_path = solver_path
        self.bufsize = int(bufsize)
        self.proc = None
        self._stop = False
        self.decisions = 50000

    def start(self, cnf_path):
        self.proc = subprocess.Popen(
            [self.solver_path, cnf_path, "-model", f"-decisions={self.decisions}", "-verb=0"],
            stdin=subprocess.PIPE,      
            stdout=subprocess.PIPE,     
            stderr=subprocess.DEVNULL,  
            bufsize=self.bufsize        
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Failed to open pipes to subprocess")

    def stop(self):
        self._stop = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def run_loop(self, callback):
        """
        Main loop:
        - read chunks from solver stdout (binary)
        - accumulate lines, parse only lines starting with b'l' or b'm'
        - when 'm done' seen, invoke callback(clauses)
        - write callback result (bytes/str) to solver stdin, flush
        - repeat until solver exits or stop() called
        """
        if self.proc is None:
            raise RuntimeError("Process not started; call start() first")

        reader = self.proc.stdout
        writer = self.proc.stdin

        # byte buffer holding partial data between reads
        buf = bytearray()
        clauses = []  # current batch of clauses: list of list[int]

        try:
            while not self._stop:
                chunk = reader.read(self.bufsize)
                if not chunk:
                    # EOF (process exited) — break loop after flushing outstanding clauses if any
                    if clauses:
                        self._handle_batch(callback, clauses, writer)
                        clauses = []
                    break

                buf.extend(chunk)
                # process all complete lines
                while True:
                    nl_idx = buf.find(b'\n')
                    if nl_idx == -1:
                        break
                    line = bytes(buf[:nl_idx])  # copy out the line (without newline)
                    del buf[:nl_idx + 1]       # remove processed line + newline

                    if not line:
                        continue

                    # check first byte to avoid a full decode for irrelevant lines
                    first = line[:1]
                    if first == b'l':
                        # expected format: b"l x1 x2 ... xn"
                        # skip leading 'l' and optional space
                        rest = line[1:].lstrip()
                        if rest:
                            # parse ints; could be negative literals
                            try:
                                # decode once per interesting line
                                lits = [int(x) for x in rest.split()]
                                clauses.append(lits)
                            except ValueError:
                                # ignore malformatted clause lines
                                continue
                        else:
                            # empty clause
                            clauses.append([])
                    elif first == b'm':
                        # check for "m done"
                        # either "m done" or other m-commands -- only act on "m done"
                        token = line[1:].lstrip()
                        if token == b'done':
                            # end of batch: send to callback
                            if clauses:
                                self._handle_batch(callback, clauses, writer)
                                clauses = []
                            else:
                                # empty batch: still call callback with empty list if desired
                                self._handle_batch(callback, [], writer)
                        else:
                            # ignore other m- lines
                            continue
                    else:
                        # ignore other lines
                        continue

            # wait for process to end (if not already)
            if self.proc.poll() is None:
                self.proc.wait()
        finally:
            # clean up
            try:
                if self.proc and self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass

    def _handle_batch(self, callback, clauses, writer):
        """
        Call the callback and write its result to writer (stdin) followed by newline (if not present).
        """
        try:
            resp = callback(clauses)
        except Exception as e:
            # callback error: write an error marker or just ignore
            # for now, write nothing and return
            return

        if resp is None:
            return

        if isinstance(resp, str):
            out = resp.encode('utf-8')
        elif isinstance(resp, bytes):
            out = resp
        else:
            # try to convert to string
            out = str(resp).encode('utf-8')

        # ensure a newline terminator (solver expects lines)
        if not out.endswith(b'\n'):
            out += b'\n'

        try:
            writer.write(out)
            writer.flush()
        except BrokenPipeError:
            # solver process closed stdin
            pass
