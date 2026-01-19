import sys, re, os, math, random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
try:
    import msvcrt
except ImportError:
    msvcrt = None

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
        m = re.match(r'([A-Z0-9_$]+)\((.+)\)$', l)
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
        m = re.match(r'([A-Z0-9_$]+)\((.+)\)', a)
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
            if i.evaluate(re.sub(r'(?<![<>!=])=(?!=)', '==', c)): return int(t)

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

class BasicInterpreter:
    """Main interpreter class managing state and execution."""
    def __init__(self):
        """Initializes the interpreter state."""
        self.lines = {}
        self.variables = {}
        self.loop_stack = []
        self.return_stack = []
        self.plt = self.fig = self.ax = None
        self.cmds = {
            'PRINT': PrintCommand(), 'LET': LetCommand(), 'INPUT': InputCommand(), 'DIM': DimCommand(),
            'GOTO': GotoCommand(), 'GOSUB': GosubCommand(), 'RETURN': ReturnCommand(), 'IF': IfCommand(),
            'FOR': ForCommand(), 'NEXT': NextCommand(), 'CLS': ClsCommand(), 'END': EndCommand(),
            'STOP': EndCommand(), 'REM': EndCommand(), 'PLOT': PlotCommand(), 'DRAW': DrawCommand(), 'CIRCLE': CircleCommand()
        }
        self.cmds['REM'].execute = lambda i, a: None
        self.reset_variables()

    def reset_variables(self):
        """Resets variables and defines standard BASIC functions."""
        self.variables = {
            'STR_STR': lambda n: (" " + str(n) if float(n) >= 0 else str(n)) if isinstance(n, (int, float)) else str(n),
            'INT': lambda n: int(float(n)), 'LEN': len, 'VAL': float, 'CHR_STR': chr, 'ASC': ord,
            'SIN': math.sin, 'COS': math.cos, 'RND': lambda n: random.random(),
            'INPUT_STR': self._input_char
        }

    def _input_char(self, n):
        """Reads n characters from input (helper for INPUT$)."""
        r = ""
        for _ in range(int(n)): r += msvcrt.getch().decode('utf-8', 'ignore') if msvcrt else sys.stdin.read(1)
        return r

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
        # Optimization: Map line numbers to indices for O(1) jumps
        self.line_map = {line: idx for idx, line in enumerate(self.sorted_lines)}
        self.pc_index = 0
        
        while self.pc_index < len(self.sorted_lines):
            ln = self.sorted_lines[self.pc_index]
            p = self.lines[ln].strip().split(" ", 1)
            cmd = self.cmds.get(p[0].upper())
            nxt = cmd.execute(self, p[1] if len(p) > 1 else "") if cmd else None
            
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
        e = re.sub(r'\b([A-Z][A-Z0-9_]*)\$', r'\1_STR', expr)
        for k, v in {'AND': ' and ', 'OR': ' or ', 'NOT': ' not ', '<>': '!='}.items():
            e = re.sub(rf'\b{k}\b' if k.isalpha() else k, v, e)
        return e

    def evaluate(self, expr):
        """Evaluates a BASIC expression."""
        py_expr = self.preprocess_expr(expr)
        try: return eval(py_expr, {}, self.variables)
        except: raise Exception("Eval Error")

if __name__ == "__main__":
    interpreter = BasicInterpreter()
    interpreter.repl()
