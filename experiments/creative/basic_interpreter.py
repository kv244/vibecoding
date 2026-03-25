import sys, re, os, math, random, time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
try:
    import msvcrt
except ImportError:
    msvcrt = None

# Compiled Regexes for Optimization
RE_LET = re.compile(r'([A-Z0-9_$]+)\s*\(\s*(.+)\s*\)$')
RE_DIM = re.compile(r'([A-Z0-9_$]+)\s*\(\s*(.+)\s*\)')
RE_IF_EQ = re.compile(r'(?<![<>!=])=(?!=)')
RE_VAR_STR = re.compile(r'\b([A-Z][A-Z0-9_]*)\$')
RE_INKEY = re.compile(r'\bINKEY_STR\b')
RE_DATA_SPLIT = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
RE_OPS_COMBINED = re.compile(r'\b(AND|OR|NOT)\b|(<>)')
RE_RENUM_GOTO = re.compile(r'\b(GOTO|GOSUB)\s+(\d+)', re.IGNORECASE)
RE_RENUM_THEN = re.compile(r'(\bTHEN\s+)(\d+)', re.IGNORECASE)

class BasicArray:
    """Array class supporting multiple dimensions."""
    def __init__(self, dims, is_str=False):
        self.dims = dims
        self.bounds = [d + 1 for d in dims]
        total_size = math.prod(self.bounds) if self.bounds else 0
        self.data = [("" if is_str else 0)] * total_size
        
        # Simplified multiplier calculation for row-major order
        self.multipliers = [1] * len(self.dims)
        for i in range(len(self.dims) - 2, -1, -1):
            self.multipliers[i] = self.multipliers[i + 1] * self.bounds[i + 1]

    def _idx(self, args):
        if len(args) != len(self.dims): raise Exception("Dimension mismatch")
        idx = 0
        for i in range(len(args)):
            idx += int(args[i]) * self.multipliers[i]
        return idx
    def __call__(self, *args): return self.data[self._idx(args)]
    def __setitem__(self, key, value): self.data[self._idx(key if isinstance(key, tuple) else (key,))] = value

class PrintCommand:
    """Handles the PRINT command."""
    def execute(self, i, a):
        """Executes the PRINT command."""
        try: print(i.evaluate(a) if a else "")
        except Exception: print("Print Error")

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
                indices = i.evaluate(idx)
                if not isinstance(indices, tuple): indices = (indices,)
                i.variables[n][indices] = v
            else:
                if l.endswith('$'): l = l[:-1] + "_STR"
                i.variables[l] = v
        except Exception: print("Assignment Error")

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
            except: val = 0
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
            try:
                dims = [int(i.evaluate(x.strip())) for x in RE_DATA_SPLIT.split(s)]
                i.variables[n] = BasicArray(dims, n.endswith("_STR"))
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
            if i.evaluate(c): return int(t)

class ForCommand:
    """Initiates a FOR loop."""
    def execute(self, i, a):
        """Executes the FOR command."""
        s = 1.0
        if "STEP" in a: a, s_str = a.split("STEP"); s = float(i.evaluate(s_str))
        if "TO" in a and "=" in a:
            v_part, e_part = a.split("TO")
            v, start = v_part.split("=")
            v = v.strip()
            i.variables[v] = float(i.evaluate(start))
            i.loop_stack.append({'v': v, 'e': float(i.evaluate(e_part)), 's': s, 'l': i.sorted_lines[i.pc_index + 1]})

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
        try:
            sec = float(i.evaluate(a))
            if i.ax and plt: plt.pause(sec)
            else: time.sleep(sec)
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

class SpriteCommand:
    """Handles sprite creation and movement (SPRITE id, x, y, [color])."""
    def execute(self, i, a):
        """Executes the SPRITE command."""
        if not i.ax: return
        parts = [x.strip() for x in a.split(',')]
        if len(parts) < 3: return
        try:
            sid = int(i.evaluate(parts[0]))
            x, y = i.evaluate(parts[1]), i.evaluate(parts[2])
            c = parts[3].strip('"') if len(parts) > 3 else 'r'
            if sid in i.sprites:
                i.sprites[sid].set_data([x], [y])
                if len(parts) > 3: i.sprites[sid].set_color(c)
            else:
                l, = i.ax.plot([x], [y], marker='o', color=c, markersize=10)
                i.sprites[sid] = l
        except Exception as e: print(f"Error: {e}")

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
        self.sprites = {}
        self.data_values = []
        self.data_ptr = 0
        self.plt = self.fig = self.ax = None
        self.cmds = {
            'PRINT': PrintCommand(), 'LET': LetCommand(), 'INPUT': InputCommand(), 'DIM': DimCommand(),
            'GOTO': GotoCommand(), 'GOSUB': GosubCommand(), 'RETURN': ReturnCommand(), 'IF': IfCommand(),
            'FOR': ForCommand(), 'NEXT': NextCommand(), 'CLS': ClsCommand(), 'END': EndCommand(),
            'STOP': EndCommand(), 'REM': EndCommand(), 'PLOT': PlotCommand(), 'DRAW': DrawCommand(), 'CIRCLE': CircleCommand(),
            'READ': ReadCommand(), 'DATA': DataCommand(), 'PAUSE': PauseCommand(), 'SPRITE': SpriteCommand()
        }
        self.cmds['REM'].execute = lambda i, a: None
        self.reset_variables()

    def reset_variables(self):
        """Resets variables and defines standard BASIC functions."""
        self.variables = {
            'STR_STR': lambda n: (" " + str(n) if float(n) >= 0 else str(n)) if isinstance(n, (int, float)) else str(n),
            'INT': lambda n: int(float(n)), 'LEN': len, 'VAL': float, 'CHR_STR': chr, 'ASC': ord,
            'SIN': math.sin, 'COS': math.cos, 'RND': lambda n: random.random(),
            'INPUT_STR': self._input_char, 'INKEY_FN': self._inkey,
            'COLLIDE': self._collide
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

    def _collide(self, id1, id2):
        """Checks if two sprites are colliding."""
        try:
            id1, id2 = int(id1), int(id2)
            if id1 not in self.sprites or id2 not in self.sprites: return 0
            x1, y1 = self.sprites[id1].get_data()
            x2, y2 = self.sprites[id2].get_data()
            dist = math.sqrt((x1[0] - x2[0])**2 + (y1[0] - y2[0])**2)
            return 1 if dist < 15 else 0
        except: return 0

    def renum(self):
        """Renumbers program lines using regex for robustness."""
        old_lines = sorted(self.lines.keys())
        if not old_lines: return
        mapping = {old: 10 + i * 10 for i, old in enumerate(old_lines)}
        
        def replace_goto(m):
            target = int(m.group(2))
            return f"{m.group(1)} {mapping.get(target, target)}"

        def replace_then(m):
            target = int(m.group(2))
            return f"{m.group(1)}{mapping.get(target, target)}"

        new_lines = {}
        for old_line_num in old_lines:
            line = self.lines[old_line_num]
            line = RE_RENUM_GOTO.sub(replace_goto, line)
            line = RE_RENUM_THEN.sub(replace_then, line)
            new_lines[mapping[old_line_num]] = line
        self.lines = new_lines

    def save_program(self, filename):
        """Saves the current program to a file."""
        try:
            with open(filename, 'w') as f:
                for line_num in sorted(self.lines.keys()):
                    f.write(f"{line_num} {self.lines[line_num]}\n")
            print(f"Saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")

    def load_program(self, filename):
        """Loads a program from a file."""
        try:
            with open(filename, 'r') as f:
                self.lines.clear()
                self.reset_variables()
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(" ", 1)
                    if parts[0].isdigit() and len(parts) > 1:
                        self.lines[int(parts[0])] = parts[1]
                print(f"Loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: File not found - {filename}")
        except Exception as e:
            print(f"Error loading file: {e}")

    def load_image(self, filename):
        """Loads and displays a JPEG image."""
        try:
            img = mpimg.imread(filename)
            if not self.ax: # Create plot if it doesn't exist
                self.fig, self.ax = plt.subplots(); self.ax.set_xlim(0, 320); self.ax.set_ylim(200, 0); self.ax.set_aspect('equal')
            self.ax.imshow(img, extent=[0, 320, 200, 0])
            if self.fig: plt.show(block=False); plt.pause(0.01)
        except FileNotFoundError: print(f"Error: Image file not found - {filename}")
        except Exception as e: print(f"Error loading image: {e}")

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
        if c == "LIST":
            for k in sorted(self.lines):
                print(f"{k} {self.lines[k]}")
        elif c == "NEW": self.lines.clear(); self.reset_variables(); plt.close('all') if plt else None
        elif c == "RUN": self.run()
        elif c == "CLS": os.system('cls' if os.name == 'nt' else 'clear')
        elif c == "RENUM": self.renum()
        elif c == "SAVE":
            match = re.search(r'"([^"]+)"', l)
            if match: self.save_program(match.group(1))
            else: print("?SYNTAX ERROR")
        elif c == "LOAD":
            match = re.search(r'"([^"]+)"', l)
            if match:
                filename = match.group(1)
                if filename.lower().endswith(('.jpg', '.jpeg')): self.load_image(filename)
                else: self.load_program(filename)
            else: print("?SYNTAX ERROR")
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
        self.sprites = {}
        
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
        def replace_ops(m): return f" {m.group(1).lower()} " if m.group(1) else "!="

        if '"' not in expr:
            e = expr
            e = RE_VAR_STR.sub(r'\1_STR', e)
            e = RE_INKEY.sub('INKEY_FN()', e)
            e = RE_OPS_COMBINED.sub(replace_ops, e)
            e = RE_IF_EQ.sub('==', e)
            return e

        tokens = re.split(r'("[^"]*")', expr)
        result = []
        for idx, token in enumerate(tokens):
            if idx % 2 == 1: result.append(token)
            else:
                e = token
                e = RE_VAR_STR.sub(r'\1_STR', e)
                e = RE_INKEY.sub('INKEY_FN()', e)
                e = RE_OPS_COMBINED.sub(replace_ops, e)
                e = RE_IF_EQ.sub('==', e)
                result.append(e)
        return "".join(result)

    def evaluate(self, expr):
        """Evaluates a BASIC expression."""
        py_expr = self.preprocess_expr(expr)
        try: return eval(py_expr, {}, self.variables)
        except Exception: raise Exception("Eval Error")

if __name__ == "__main__":
    interpreter = BasicInterpreter()
    interpreter.repl()
