import sys
import re
import os
import math
import msvcrt


class BasicList(list):
    """Allows list access via parentheses to mimic BASIC arrays in eval()"""
    def __call__(self, index):
        idx = int(index)
        if idx < 0 or idx >= len(self):
            raise IndexError("Variable size exceeded")
        return self[idx]

class VariablesDict(dict):
    """Dictionary that returns default values for missing keys"""
    def __getitem__(self, key):
        if key not in self:
            # Only provide defaults for BASIC variables (Uppercase)
            # This allows Python builtins like 'str', 'int' to pass through to globals/builtins
            if key and key[0].isupper():
                # Raise error for missing variable access (User Request)
                # This will interrupt eval()
                raise KeyError("Variable not found")
        return super().__getitem__(key)

class BasicInterpreter:
    def __init__(self):
        self.lines = {}  # {line_number: code_string}
        self.variables = VariablesDict()  # {var_name: value}
        self.loop_stack = [] # [(var_name, end_val, step_val, line_body_start)]
        self.return_stack = [] # [return_line_number]
        self.running = False
        
        self.reset_variables()

    def reset_variables(self):
        self.variables.clear()
        # Standard Functions
        self.variables['STR_STR'] = str
        self.variables['INT'] = int
        self.variables['LEN'] = len
        self.variables['VAL'] = float # or int depending
        self.variables['CHR_STR'] = chr
        self.variables['ASC'] = ord
        self.variables['SIN'] = math.sin
        self.variables['COS'] = math.cos
        # Special handling for INPUT$ is done via method binding or custom eval
        # Since our eval uses self.variables, we can map INPUT_STR to a function
        self.variables['INPUT_STR'] = self._input_char

    def _input_char(self, n):
        """Read n characters from console without echo (like GET$)"""
        res = ""
        count = int(n)
        for _ in range(count):
            if msvcrt:
                # msvcrt.getch() returns bytes
                char = msvcrt.getch().decode('utf-8', 'ignore')
                res += char
            else:
                # Fallback for non-windows (though user is on windows)
                # This is blocking line input effectively
                res += sys.stdin.read(1)
        return res

    def repl(self):
        print("Python Toy BASIC Interpreter (Advanced)")
        print("Commands: LIST, RUN, NEW, BYE, CLS")
        
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    continue
                    
                parts = line.split(" ", 1)
                
                # Check for line number
                if parts[0].isdigit():
                    line_num = int(parts[0])
                    if len(parts) > 1:
                        self.lines[line_num] = parts[1]
                    else:
                        # Delete line if only number provided
                        if line_num in self.lines:
                            del self.lines[line_num]
                else:
                    self.execute_immediate(parts[0].upper(), line)
                    
            except KeyboardInterrupt:
                print("\nInterrupted")
            except Exception as e:
                print(f"Error: {e}")

    def execute_immediate(self, command, full_line):
        if command == "LIST":
            for lineno in sorted(self.lines.keys()):
                print(f"{lineno} {self.lines[lineno]}")
        elif command == "NEW":
            self.lines.clear()
            self.reset_variables()
            
            self.loop_stack.clear()
            self.return_stack.clear()
            print("Program Cleared")
        elif command == "RUN":
            self.run_program()
        elif command == "CLS":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif command == "BYE" or command == "EXIT":
            sys.exit(0)
        else:
            print(f"Unknown command: {command}")

    def run_program(self):
        if not self.lines:
            return

        sorted_lines = sorted(self.lines.keys())
        self.reset_variables()
        self.loop_stack.clear() # Reset loops on run
        self.return_stack.clear()
        pc_index = 0
        
        while pc_index < len(sorted_lines):
            line_num = sorted_lines[pc_index]
            code = self.lines[line_num]
            
            # Simple parsing: CMD args
            parts = code.strip().split(" ", 1)
            cmd = parts[0].upper()
            args = parts[1] if len(parts) > 1 else ""
            
            # Execute
            next_line_num = self.execute_statement(cmd, args, line_num, sorted_lines, pc_index)
            
            if next_line_num is not None:
                # GOTO / Branch happened
                if next_line_num == -1: # End
                    break
                    
                # Find index of this line number
                try:
                    pc_index = sorted_lines.index(next_line_num)
                except ValueError:
                    print(f"Runtime Error: Line {next_line_num} not found")
                    break
            else:
                # Normal flow
                pc_index += 1

    def preprocess_expr(self, expr):
        """Converts BASIC syntax to Python syntax for eval"""
        # Replace string vars "A$" with "A_STR"
        # Look for letter followed by $
        expr = re.sub(r'\b([A-Z][A-Z0-9]*)\$', r'\1_STR', expr)
        # Operators (BASIC -> Python)
        # Use regex to avoid replacing inside strings (simple approach: assume keywords are outside quotes? 
        # But this function doesn't parse quotes. 
        # Better: match word boundaries \bAND\b
        expr = re.sub(r'\bAND\b', ' and ', expr)
        expr = re.sub(r'\bOR\b', ' or ', expr)
        expr = expr.replace("<>", "!=")
        # Note: single '=' for equality is handled in IF specific logic usually,
        # but for boolean exprs in general we might want '=='
        # For simplicity, we assume IF handles '=' fixup, and expressions are mostly math
        return expr

    def evaluate(self, expr):
        py_expr = self.preprocess_expr(expr)
        try:
            return eval(py_expr, {}, self.variables)
        except KeyError as e:
            # Variable not found
            # The exception args will be ('Variable not found',) or just 'VARNAME' if implicit?
            # Our VariablesDict raises KeyError("Variable not found")
            if str(e) == "'Variable not found'":
                raise Exception("Variable not found")
            else:
                # If eval raises KeyError for other reasons, it might be looking up the name
                # If we access variables[missing], it raises.
                # Standard Python KeyError repr includes quotes.
                raise Exception("Variable not found")
        except IndexError as e:
            raise Exception(str(e)) # "Variable size exceeded"
        except NameError as e:
            # Python raises NameError when lookup fails, even if __getitem__ raised KeyError
            raise Exception("Variable not found")
        except Exception as e:
            # print(f"Eval Error: {e} in '{expr}' -> '{py_expr}'")
            raise e

    def execute_statement(self, cmd, args, current_lineno, sorted_lines, pc_index):
        if cmd == "PRINT":
            if args.startswith('"') and args.endswith('"'):
                print(args[1:-1])
            elif not args:
                print("")
            else:
                try:
                    print(self.evaluate(args))
                except Exception as e:
                    print(f"Print Error: {e}")

        elif cmd == "LET":
            if "=" not in args:
                print("Syntax Error: LET var = expr")
                return None
            
            lhs, rhs = args.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # Check for Array Assignment: A(idx) = val or A$(idx) = val
            array_match = re.match(r'^([A-Z][A-Z0-9]*\$?)\((.+)\)$', lhs)
            
            try:
                val = self.evaluate(rhs)
                
                if array_match:
                    # Array assignment
                    arr_name = array_match.group(1)
                    if arr_name.endswith('$'):
                        arr_name = arr_name[:-1] + "_STR"
                    
                    idx_expr = array_match.group(2)
                    idx = int(self.evaluate(idx_expr))
                    
                    if arr_name in self.variables and isinstance(self.variables[arr_name], list):
                        self.variables[arr_name][idx] = val
                    else:
                        print(f"Runtime Error: Array {arr_name} not defined")
                
                else:
                    # Scalar assignment
                    # Handle string var names
                    if lhs.endswith('$'):
                        lhs = lhs[:-1] + "_STR"
                    self.variables[lhs] = val
            except Exception as e:
                print(f"LET Error: {e}")

        elif cmd == "INPUT":
            # INPUT ["Prompt";] VAR, VAR2, ...
            
            # Check for prompt
            args = args.strip()
            prompt = "? "
            
            # Simple parser for Prompt string
            # Look for " at start
            if args.startswith('"'):
                # Find closing quote
                end_quote = args.find('"', 1)
                if end_quote != -1:
                    prompt = args[1:end_quote]
                    # Check for semicolon or comma separator
                    rest = args[end_quote+1:].strip()
                    if rest.startswith(';') or rest.startswith(','):
                        args = rest[1:].strip()
            
            # Split remaining args by comma
            # Note: This simple split breaks if array indices contain commas e.g. A(1,2)
            # For now, assume simple vars or simple arrays A(1)
            var_list = [v.strip() for v in args.split(',') if v.strip()]
            
            for i, var_name in enumerate(var_list):
                # Use prompt only for first variable? Or prompt for all?
                # Standard BASIC: Prompt for first, "?" for others? Or just wait?
                # Let's print prompt for first
                current_prompt = prompt if i == 0 else "? "
                
                print(current_prompt, end='', flush=True)
                
                try:
                    val_str = input()
                except EOFError:
                    val_str = ""

                # Try to convert to float/int, else string
                try:
                    if "." in val_str:
                        val = float(val_str)
                    else:
                        val = int(val_str)
                except ValueError:
                    val = val_str
                
                if var_name.endswith('$'):
                    var_name = var_name[:-1] + "_STR"
                    
                self.variables[var_name] = val

        elif cmd == "DIM":
            # DIM A(10) or DIM A$(10)
            match = re.match(r'([A-Z][A-Z0-9]*\$?)\((.+)\)', args)
            if match:
                name = match.group(1)
                if name.endswith('$'):
                    name = name[:-1] + "_STR"
                
                size_expr = match.group(2)
                try:
                    size = int(self.evaluate(size_expr))
                    # +1 to include the index itself (0..size)
                    initial_value = "" if name.endswith("_STR") else 0
                    self.variables[name] = BasicList([initial_value] * (size + 1))
                except Exception as e:
                    print(f"DIM Error: {e}")
            else:
                print("Syntax Error in DIM")

        elif cmd == "GOTO":
            try:
                return int(args)
            except ValueError:
                print("Syntax Error: GOTO <linenum>")

        elif cmd == "GOSUB":
            try:
                target_line = int(args)
                # Calculate return line (line after current GOSUB)
                return_index = pc_index + 1
                if return_index < len(sorted_lines):
                    return_line = sorted_lines[return_index]
                    self.return_stack.append(return_line)
                    return target_line
                else:
                    print("Error: GOSUB at end of program (no return point)")
                    return -1
            except ValueError:
                print("Syntax Error: GOSUB <linenum>")

        elif cmd == "RETURN":
            if not self.return_stack:
                print("Error: RETURN without GOSUB")
                return -1
            
            return self.return_stack.pop()

        elif cmd == "IF":
            if "THEN" not in args:
                print("Syntax Error: IF <cond> THEN <line>")
                return None
            
            cond_str, target_line = args.split("THEN", 1)
            
            # Fix equality operator for eval
            if "=" in cond_str and "==" not in cond_str and "<=" not in cond_str and ">=" not in cond_str and "!=" not in cond_str:
                 cond_str = cond_str.replace("=", "==")

            try:
                if self.evaluate(cond_str):
                    return int(target_line)
            except Exception as e:
                print(f"Condition Error: {e}")

        elif cmd == "FOR":
            # FOR I = 1 TO 10 [STEP 2]
            # 1. Parse
            try:
                # Remove FOR
                # Format: VAR = START TO END [STEP S]
                
                step_val = 1
                if "STEP" in args:
                    main_part, step_part = args.split("STEP")
                    step_val = int(self.evaluate(step_part))
                else:
                    main_part = args
                
                if "TO" not in main_part or "=" not in main_part:
                    print("Syntax Error: FOR VAR = Start TO End [STEP]")
                    return None
                    
                assign_part, end_part = main_part.split("TO")
                var_name, start_expr = assign_part.split("=")
                var_name = var_name.strip()
                
                start_val = int(self.evaluate(start_expr))
                end_val = int(self.evaluate(end_part))
                
                # Assign initial value
                self.variables[var_name] = start_val
                
                # Push to stack
                # Find line number of the NEXT statement? No, NEXT jumps back here?
                # Actually standard BASIC: FOR loop pushes return address (line *after* FOR)
                # But here we loop efficiently if we store the definition
                
                next_stmt_idx = pc_index + 1
                if next_stmt_idx < len(sorted_lines):
                    next_stmt_lineno = sorted_lines[next_stmt_idx]
                else:
                    print("Error: For loop at end of program")
                    return -1

                self.loop_stack.append({
                    'var': var_name,
                    'end': end_val,
                    'step': step_val,
                    'body_line': next_stmt_lineno
                })
                
            except Exception as e:
                print(f"FOR Error: {e}")

        elif cmd == "NEXT":
            var_name = args.strip()
            if not self.loop_stack:
                print("Error: NEXT without FOR")
                return None
            
            # Get current loop
            # BASIC typically requires proper nesting, check top of stack
            current_loop = self.loop_stack[-1]
            if current_loop['var'] != var_name:
                print(f"Error: NEXT {var_name} doesn't match FOR {current_loop['var']}")
                return None
            
            # Increment
            step = current_loop['step']
            val = self.variables[var_name]
            val += step
            self.variables[var_name] = val
            
            # Check condition
            # If step > 0: loop while val <= end
            # If step < 0: loop while val >= end
            should_loop = False
            if step > 0:
                if val <= current_loop['end']:
                    should_loop = True
            else:
                if val >= current_loop['end']:
                    should_loop = True
            
            if should_loop:
                return current_loop['body_line']
            else:
                self.loop_stack.pop()
                # Fall through to next line

        elif cmd == "CLS":
            os.system('cls' if os.name == 'nt' else 'clear')

        elif cmd == "END" or cmd == "STOP":
            return -1

        return None

if __name__ == "__main__":
    interpreter = BasicInterpreter()
    interpreter.repl()
