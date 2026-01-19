import sys, re, os, math, random, time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
try:
    import msvcrt
except ImportError:
    msvcrt = None

# Compiled Regexes for Optimization
RE_LET = re.compile(r'([A-Z0-9_$]+)\((.+)\)$')
RE_DIM = re.compile(r'([A-Z0-9_$]+)\((.+)\)')
RE_IF_EQ = re.compile(r'(?<![<>!=])=(?!=)')
RE_VAR_STR = re.compile(r'\b([A-Z][A-Z0-9_]*)\$')
RE_INKEY = re.compile(r'\bINKEY_STR\b')
RE_DATA_SPLIT = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
RE_OPS = [(re.compile(r'\bAND\b'), ' and '), (re.compile(r'\bOR\b'), ' or '), 
          (re.compile(r'\bNOT\b'), ' not '), (re.compile(r'<>'), '!=')]

class BasicList(list):
    """List subclass allowing access via () for BASIC array compatibility."""
    def __call__(self, i): return self[int(i)]

class PrintCommand:
    """Handles the PRINT command."""
    def execute(self, i, a):
        """Executes the PRINT command."""
        try: print(i.evaluate(a) if a else "")
        except Exception as e: print(f"Error: {e}")

class LetCommand:
    """Handles variable assignment (LET)."""
    def execute(self, i, a):
        """Executes the LET command."""
        if "=" not in a: return
        l, r = a.split("=", 1)
        l, r = l.strip(), r.strip()
        m = RE_LET.match(l)
        try:
            v = i.evaluate(r)
            if m:
                n, idx = m.groups()
                if n.endswith('$'): n = n[:-1] + "_STR"
                i.variables[n][int(i.evaluate(idx))] = v
            else:
                if l.endswith('$'): l = l[:-1] + "_STR"
                i.variables[l] = v
        except Exception as e: print(f"Error: {e}")

class InputCommand:
    """Handles user input (INPUT)."""
    def execute(self, i, a):
        """Executes the INPUT command."""
        p = "? "
        if a.startswith('"'):
            e = a.find('"', 1)
            if e > 0: p, a = a[1:e], a[e+1:].lstrip(";,").strip()
        for idx, v in enumerate([x.strip() for x in a.split(',') if x.strip()]):
            print(p if idx == 0 else "? ", end='', flush=True)
            val = input()
            try: val = float(val) if '.' in val else int(val)
            except: pass
            if v.endswith('$'): v = v[:-1] + "_STR"
            i.variables[v] = val

class DimCommand:
    """Handles array dimensioning (DIM)."""
    def execute(self, i, a):
        """Executes the DIM command."""
        m = RE_DIM.match(a)
        if m:
            n, s = m.groups()
            if n.endswith('$'): n = n[:-1] + "_STR"
            try: i.variables[n] = BasicList([("" if n.endswith("_STR") else 0)] * (int(i.evaluate(s)) + 1))
            except Exception as e: print(f"Error: {e}")

class GotoCommand:
    """Handles unconditional jumps (GOTO)."""
    def execute(self, i, a): return int(a)

class GosubCommand:
    """Handles subroutine calls (GOSUB)."""
    def execute(self, i, a):
        """Executes the GOSUB command."""
        i.return_stack.append(i.sorted_lines[i.pc_index + 1])
        return int(a)

class ReturnCommand:
    """Returns from a subroutine (RETURN)."""
    def execute(self, i, a): return i.return_stack.pop() if i.return_stack else -1

class IfCommand:
    """Handles conditional logic (IF ... THEN)."""
    def execute(self, i, a):
        """Executes the IF command."""
        if "THEN" in a:
            c, t = a.split("THEN", 1)
            if i.evaluate(RE_IF_EQ.sub('==', c)): return int(t)

class ForCommand:
    """Initiates a FOR loop."""
    def execute(self, i, a):
        """Executes the FOR command."""
        s = 1
        if "STEP" in a: a, s_str = a.split("STEP"); s = int(i.evaluate(s_str))
        if "TO" in a and "=" in a:
            v_part, e_part = a.split("TO")
            v, start = v_part.split("=")
            v = v.strip()
            i.variables[v] = int(i.evaluate(start))
            i.loop_stack.append({'v': v, 'e': int(i.evaluate(e_part)), 's': s, 'l': i.sorted_lines[i.pc_index + 1]})

class NextCommand:
    """Ends a FOR loop step (NEXT)."""
    def execute(self, i, a):
        """Executes the NEXT command."""
        v = a.strip()
        if i.loop_stack:
            l = i.loop_stack[-1]
            if l['v'] == v:
                i.variables[v] += l['s']
                if (l['s'] > 0 and i.variables[v] <= l['e']) or (l['s'] < 0 and i.variables[v] >= l['e']): return l['l']
                i.loop_stack.pop()

class ClsCommand:
    """Clears the console screen."""
    def execute(self, i, a):
        os.system('cls' if os.name == 'nt' else 'clear')
        if i.ax:
            i.ax.cla() # Also clear the plot if it exists

class EndCommand:
    """Terminates program execution."""
    def execute(self, i, a): return -1

class PauseCommand:
    """Pauses execution for a specified duration (PAUSE)."""
    def execute(self, i, a):
        """Executes the PAUSE command."""
        try: time.sleep(float(i.evaluate(a)))
        except Exception as e: print(f"Error: {e}")

class PlotCommand:
    """Plots a single point on the graphics window."""
    def execute(self, i, a):
        """Executes the PLOT command."""
        if i.ax:
            x, y = map(i.evaluate, a.split(','))
            i.ax.plot([x], [y], 'k.', markersize=1)
            i.last_plot_pos = (x, y)

class DrawCommand:
    """Draws a line from the last point to a new point."""
    def execute(self, i, a):
        """Executes the DRAW command."""
        if i.ax:
            x, y = map(i.evaluate, a.split(','))
            lx, ly = i.last_plot_pos
            i.ax.plot([lx, x], [ly, y], 'k-', linewidth=1)
            i.last_plot_pos = (x, y)

class CircleCommand:
    """Draws a circle."""
    def execute(self, i, a):
        """Executes the CIRCLE command."""
        if i.ax:
            x, y, r = map(i.evaluate, a.split(','))
            i.ax.add_patch(Circle((x, y), r, color='k', fill=False))

class ReadCommand:
    """Reads values from DATA statements (READ)."""
    def execute(self, i, a):
        """Executes the READ command."""
        for v in [x.strip() for x in a.split(',') if x.strip()]:
            if i.data_ptr >= len(i.data_values):
                print("Error: Out of data")
                return -1
            val = i.data_values[i.data_ptr]
            i.data_ptr += 1
            if v.endswith('$'): i.variables[v[:-1] + "_STR"] = val
            else:
                try: i.variables[v] = float(val) if '.' in val else int(val)
                except: print(f"Error: expected number for {v}"); return -1

class DataCommand:
    """Defines data values (DATA)."""
    def execute(self, i, a): pass

class BasicInterpreter:
    """Main interpreter class managing state and execution."""
    def __init__(self):
        """Initializes the interpreter state."""
        self.lines = {}
        self.variables = {}
        self.loop_stack = []
        self.return_stack = []
        self.data_values = []
        self.data_ptr = 0
        self.plt = self.fig = self.ax = None
        self.cmds = {
            'PRINT': PrintCommand(), 'LET': LetCommand(), 'INPUT': InputCommand(), 'DIM': DimCommand(),
            'GOTO': GotoCommand(), 'GOSUB': GosubCommand(), 'RETURN': ReturnCommand(), 'IF': IfCommand(),
            'FOR': ForCommand(), 'NEXT': NextCommand(), 'CLS': ClsCommand(), 'END': EndCommand(),
            'STOP': EndCommand(), 'REM': EndCommand(), 'PLOT': PlotCommand(), 'DRAW': DrawCommand(), 'CIRCLE': CircleCommand(),
            'READ': ReadCommand(), 'DATA': DataCommand(), 'PAUSE': PauseCommand()
        }
        self.cmds['REM'].execute = lambda i, a: None
        self.reset_variables()

    def reset_variables(self):
        """Resets variables and defines standard BASIC functions."""
        self.variables = {
            'STR_STR': lambda n: (" " + str(n) if float(n) >= 0 else str(n)) if isinstance(n, (int, float)) else str(n),
            'INT': lambda n: int(float(n)), 'LEN': len, 'VAL': float, 'CHR_STR': chr, 'ASC': ord,
            'SIN': math.sin, 'COS': math.cos, 'RND': lambda n: random.random(),
            'INPUT_STR': self._input_char, 'INKEY_FN': self._inkey
        }

    def _input_char(self, n):
        """Reads n characters from input (helper for INPUT$)."""
        r = ""
        for _ in range(int(n)): r += msvcrt.getch().decode('utf-8', 'ignore') if msvcrt else sys.stdin.read(1)
        return r

    def _inkey(self):
        """Checks for a key press and returns it, or empty string."""
        if msvcrt and msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8', 'ignore')
        return ""

    def renum(self):
        """Renumbers program lines."""
        old_lines = sorted(self.lines.keys())
        if not old_lines: return
        mapping = {old: 10 + i * 10 for i, old in enumerate(old_lines)}
        new_lines = {}
        for old in old_lines:
            line = self.lines[old]
            parts = line.strip().split(" ", 1)
            cmd = parts[0].upper()
            if cmd in ["GOTO", "GOSUB"] and len(parts) > 1:
                try:
                    target = int(parts[1].strip())
                    if target in mapping: line = f"{cmd} {mapping[target]}"
                except: pass
            elif cmd == "IF" and "THEN" in line:
                p, t = line.split("THEN", 1)
                try:
                    target = int(t.strip())
                    if target in mapping: line = f"{p}THEN {mapping[target]}"
                except: pass
            new_lines[mapping[old]] = line
        self.lines = new_lines

    def repl(self):
        """Read-Eval-Print Loop for immediate mode."""
        print("Toy BASIC")
        while True:
            try:
                l = input("> ").strip()
                if not l: continue
                p = l.split(" ", 1)
                if p[0].isdigit():
                    if len(p) > 1: self.lines[int(p[0])] = p[1]
                    elif int(p[0]) in self.lines: del self.lines[int(p[0])]
                else: self.exec_cmd(p[0].upper(), l)
            except KeyboardInterrupt: break
            except Exception as e: print(f"Error: {e}")

    def exec_cmd(self, c, l):
        """Executes immediate commands like RUN, LIST, NEW."""
        if c == "LIST": [print(f"{k} {self.lines[k]}") for k in sorted(self.lines)]
        elif c == "NEW": self.lines.clear(); self.reset_variables(); plt.close('all') if plt else None
        elif c == "RUN": self.run()
        elif c == "CLS": os.system('cls' if os.name == 'nt' else 'clear')
        elif c == "RENUM": self.renum()
        elif c in ["BYE", "EXIT"]: sys.exit(0)
        else: print("?")

    def run(self):
        """Executes the stored program."""
        if not self.lines:
            return
        if plt:
            plt.close('all')
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, 320)
            self.ax.set_ylim(200, 0)
            self.ax.set_aspect('equal')
        
        self.reset_variables()
        self.loop_stack.clear()
        self.return_stack.clear()
        self.sorted_lines = sorted(self.lines)
        
        self.data_values = []
        self.data_ptr = 0
        self.program = [] # Pre-parsed program: list of (command, arg)

        for ln in self.sorted_lines:
            p = self.lines[ln].strip().split(" ", 1)
            cmd_name = p[0].upper()
            arg = p[1] if len(p) > 1 else ""
            if cmd_name == "DATA":
                for x in RE_DATA_SPLIT.split(arg):
                    x = x.strip()
                    self.data_values.append(x[1:-1] if x.startswith('"') and x.endswith('"') else x)
            self.program.append((self.cmds.get(cmd_name), arg))

        # Optimization: Map line numbers to indices for O(1) jumps
        self.line_map = {line: idx for idx, line in enumerate(self.sorted_lines)}
        self.pc_index = 0
        
        n_lines = len(self.program)
        while self.pc_index < n_lines:
            cmd, arg = self.program[self.pc_index]
            nxt = cmd.execute(self, arg) if cmd else None
            
            if nxt == -1:
                break
            if nxt is not None:
                if nxt in self.line_map:
                    self.pc_index = self.line_map[nxt]
                else:
                    print(f"Error: Line {nxt} not found.")
                    break
            else: self.pc_index += 1
        if self.fig: plt.show()

    def preprocess_expr(self, expr):
        """Preprocesses BASIC expressions into Python syntax."""
        # Fix: Include underscore in variable names and return the modified string 'e'
        e = RE_VAR_STR.sub(r'\1_STR', expr)
        e = RE_INKEY.sub('INKEY_FN()', e)
        for p, r in RE_OPS: e = p.sub(r, e)
        return e

    def evaluate(self, expr):
        """Evaluates a BASIC expression."""
        py_expr = self.preprocess_expr(expr)
        try: return eval(py_expr, {}, self.variables)
        except: raise Exception("Eval Error")

if __name__ == "__main__":
    interpreter = BasicInterpreter()
    interpreter.repl()
