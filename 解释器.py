import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, simpledialog, ttk
import re
import os
import sys
from io import StringIO
from typing import Dict, Any, List, Union, Optional, Tuple, Callable

class BiangIDE:
    def __init__(self, root):
        self.root = root
        self.root.title("Biang语言++ IDE")
        self.root.geometry("1000x800")
        
        self._init_menu()
        self._init_ui()
        self.interpreter = BiangInterpreter(self.root)
        self.current_file = None
    
    def _init_menu(self):
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="新建", command=self.new_file)
        file_menu.add_command(label="打开", command=self.open_file)
        file_menu.add_command(label="保存", command=self.save_file)
        file_menu.add_command(label="另存为", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 运行菜单
        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="运行", command=self.run_code)
        run_menu.add_command(label="停止", command=self.stop_execution)
        menubar.add_cascade(label="运行", menu=run_menu)
        
        self.root.config(menu=menubar)
    
    def _init_ui(self):
        # 主面板
        main_panel = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)
        
        # 代码编辑区
        self.code_text = scrolledtext.ScrolledText(
            main_panel, wrap=tk.WORD, font=('Consolas', 12)
        )
        self.code_text.pack(fill=tk.BOTH, expand=True)
        
        # 输出区
        self.output_text = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=('Consolas', 12), state='disabled'
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
    
    def new_file(self):
        self.code_text.delete("1.0", tk.END)
        self.current_file = None
        self.clear_output()
        self.root.title("Biang语言++ IDE")
    
    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Biang Files", "*.biang"), ("All Files", "*.*")]
        )
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.code_text.delete("1.0", tk.END)
                self.code_text.insert("1.0", file.read())
            self.current_file = file_path
            self.root.title(f"Biang语言++ IDE - {os.path.basename(file_path)}")
    
    def save_file(self):
        if self.current_file:
            content = self.code_text.get("1.0", "end-1c")
            with open(self.current_file, 'w', encoding='utf-8') as file:
                file.write(content)
        else:
            self.save_as_file()
    
    def save_as_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".biang",
            filetypes=[("Biang Files", "*.biang"), ("All Files", "*.*")]
        )
        if file_path:
            content = self.code_text.get("1.0", "end-1c")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            self.current_file = file_path
            self.root.title(f"Biang语言++ IDE - {os.path.basename(file_path)}")
    
    def clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state='disabled')
    
    def write_output(self, text):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')
    
    def run_code(self):
        self.clear_output()
        code = self.code_text.get("1.0", "end-1c")
        
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            self.interpreter.interpret(code)
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            self.write_output(output)
        except Exception as e:
            self.write_output(f"错误: {str(e)}\n")
    
    def stop_execution(self):
        self.interpreter.stop()

class BiangInterpreter:
    def __init__(self, root_window):
        self.variables = {}
        self.pointers = {}
        self.functions = {}
        self.call_stack = []
        self.current_blocks = []
        self.code_lines = []
        self.current_line = 0
        self.return_value = None
        self.root_window = root_window
        self.gui_enabled = False
        self.gui_objects = {}
        self._running = False
        self.comment_pattern = re.compile(r'/\?.*?$|/\*.*?\*/', re.DOTALL | re.MULTILINE)
    
    def stop(self):
        self._running = False
    
    def _remove_comments(self, code: str) -> str:
        return self.comment_pattern.sub('', code)
    
    def interpret(self, code: str):
        if not code.startswith('<?biang') or not code.endswith('biang?>'):
            raise SyntaxError("代码必须以'<?biang'开头并以'biang?>'结尾")
        
        code = self._remove_comments(code[7:-7].strip())
        self.code_lines = [line.strip() for line in code.split('\n') if line.strip()]
        self.current_line = 0
        self._reset_state()
        
        while self._running and self.current_line < len(self.code_lines):
            line = self.code_lines[self.current_line]
            self._process_line(line)
            self.current_line += 1
    
    def _reset_state(self):
        self.variables = {}
        self.pointers = {}
        self.functions = {}
        self.call_stack = []
        self.current_blocks = []
        self.return_value = None
        self.gui_enabled = False
        self.gui_objects = {}
        self._running = True
    
    def _process_line(self, line: str):
        try:
            if not line or line.startswith('/?'):
                return
            
            if line == "including gui":
                self.gui_enabled = True
                return
            
            if line.startswith('ptr '):
                self._handle_pointer(line[4:])
            elif line.startswith('*'):
                self._handle_dereference(line)
            elif line.startswith('var '):
                self._handle_var_declaration(line[4:])
            elif line.startswith('list '):
                self._handle_list_declaration(line[5:])
            elif line.startswith('func '):
                self._handle_function_declaration(line[5:])
            elif line.startswith('if '):
                self._handle_if(line[3:])
            elif line == 'else':
                self._handle_else()
            elif line.startswith('loop '):
                self._handle_loop(line[5:])
            elif line.startswith('while '):
                self._handle_while(line[6:])
            elif line.startswith('Print(') or line.startswith('PrintLn('):
                self._handle_print(line)
            elif line.startswith('scans('):
                self._handle_input(line)
            elif '(' in line and line.endswith(')'):
                func_name = line.split('(', 1)[0]
                if func_name in self.functions:
                    self._handle_function_call(line)
            elif '=' in line and not any(kw in line for kw in ['var ', 'list ', 'func ', 'if ', 'loop ', 'while ']):
                self._handle_assignment(line)
            elif line == '<?':
                self._handle_block_start()
            elif line == '?>':
                self._handle_block_end()
            elif self.gui_enabled and line.startswith("var Window"):
                self._handle_window_declaration(line)
            elif self.gui_enabled and line.startswith("var Button"):
                self._handle_button_declaration(line)
            elif self.gui_enabled and line.startswith("var Label"):
                self._handle_label_declaration(line)
            else:
                raise SyntaxError(f"无法识别的语句: {line}")
        except Exception as e:
            raise RuntimeError(f"行 {self.current_line+1}: {str(e)}")
    
    def _handle_pointer(self, line: str):
        match = re.match(r'(\w+)\s*\*(\w+)\s*=\s*&(\w+)', line)
        if not match:
            raise SyntaxError("指针声明语法错误")
        
        var_type, ptr_name, target_var = match.groups()
        if target_var not in self.variables:
            raise NameError(f"未定义变量: {target_var}")
        
        self.pointers[ptr_name] = {
            'type': var_type,
            'target': target_var
        }
    
    def _handle_dereference(self, line: str):
        if '=' in line:
            left, right = line.split('=', 1)
            left = left.strip()
            right = right.strip()
            
            if left.startswith('*'):
                ptr_name = left[1:]
                if ptr_name not in self.pointers:
                    raise NameError(f"未定义指针: {ptr_name}")
                
                value = self._evaluate_expression(right)
                self.variables[self.pointers[ptr_name]['target']] = value
            elif right.startswith('*'):
                ptr_name = right[1:]
                if ptr_name not in self.pointers:
                    raise NameError(f"未定义指针: {ptr_name}")
                
                self.variables[left] = self.variables[self.pointers[ptr_name]['target']]
        else:
            raise SyntaxError("指针解引用语法错误")
    
    def _handle_if(self, line: str):
        match = re.match(r'\{(.+)\}\s*<\?', line)
        if not match:
            raise SyntaxError("if语句语法错误")
        
        condition = self._evaluate_condition(match.group(1))
        block_end = self._find_block_end(self.current_line)
        
        has_else = False
        else_line = -1
        if block_end + 1 < len(self.code_lines) and self.code_lines[block_end + 1].strip() == 'else':
            has_else = True
            else_line = block_end + 1
            block_end = self._find_block_end(else_line)
        
        self.current_blocks.append({
            'type': 'if',
            'condition': condition,
            'start_line': self.current_line,
            'end_line': block_end,
            'has_else': has_else,
            'else_line': else_line
        })
        
        if not condition:
            self.current_line = else_line if has_else else block_end
    
    def _handle_else(self):
        if not self.current_blocks or self.current_blocks[-1]['type'] != 'if':
            raise SyntaxError("else没有对应的if")
        
        current_block = self.current_blocks[-1]
        if not current_block['has_else']:
            raise SyntaxError("if语句没有对应的else")
        
        if current_block['condition']:
            self.current_line = current_block['end_line']
    
    def _handle_var_declaration(self, line: str):
        match = re.match(r'(\w+)\s+(\w+)\s*=\s*(.+)', line)
        if not match:
            raise SyntaxError(f"变量声明语法错误: {line}")
        
        var_type, var_name, value = match.groups()
        value = self._parse_value(value, var_type)
        self.variables[var_name] = value
    
    def _handle_list_declaration(self, line: str):
        match = re.match(r'(\w+)\s+(\w+)\s*=\s*\{(.+)\}', line)
        if not match:
            raise SyntaxError(f"列表声明语法错误: {line}")
        
        list_type, list_name, items_str = match.groups()
        items = [item.strip() for item in items_str.split(',')]
        
        converted_items = []
        for item in items:
            converted_items.append(self._parse_value(item, list_type))
        
        self.variables[list_name] = converted_items
    
    def _handle_function_declaration(self, line: str):
        match = re.match(r'(\w+)\(([^)]*)\)\s*<\?', line)
        if not match:
            raise SyntaxError(f"函数定义语法错误: {line}")
        
        func_name, params_str = match.groups()
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        
        return_type = 'void'
        func_params = []
        
        for param in params:
            if param.startswith('returntype='):
                return_type = param.split('=', 1)[1]
            else:
                param_name, param_type = param.split(':', 1)
                func_params.append((param_name.strip(), param_type.strip()))
        
        self.functions[func_name] = {
            'return_type': return_type,
            'params': func_params,
            'start_line': self.current_line,
            'end_line': self._find_block_end(self.current_line)
        }
        
        self.current_line = self.functions[func_name]['end_line']
    
    def _handle_loop(self, line: str):
        match = re.match(r'\((\d+)\)\s*<\?', line)
        if not match:
            raise SyntaxError(f"循环语法错误: {line}")
        
        iterations = int(match.group(1))
        block_end = self._find_block_end(self.current_line)
        
        self.current_blocks.append({
            'type': 'loop',
            'iterations': iterations,
            'current_iteration': 0,
            'start_line': self.current_line,
            'end_line': block_end
        })
        
        self.current_blocks[-1]['current_iteration'] += 1
    
    def _handle_while(self, line: str):
        match = re.match(r'\((.+)\)\s*<\?', line)
        if not match:
            raise SyntaxError(f"while循环语法错误: {line}")
        
        condition = match.group(1)
        block_end = self._find_block_end(self.current_line)
        
        condition_met = self._evaluate_condition(condition)
        
        self.current_blocks.append({
            'type': 'while',
            'condition': condition,
            'condition_met': condition_met,
            'start_line': self.current_line,
            'end_line': block_end
        })
        
        if not condition_met:
            self.current_line = block_end
    
    def _handle_print(self, line: str):
        is_println = line.startswith('PrintLn(')
        match = re.match(r'Print(?:Ln)?\((.+)\)', line)
        if not match:
            raise SyntaxError(f"输出语句语法错误: {line}")
        
        arg = match.group(1)
        value = self._evaluate_expression(arg)
        
        if is_println:
            print(value)
        else:
            print(value, end='')
    
    def _handle_input(self, line: str):
        match = re.match(r'scans\(\s*"?(.+?)"?\s*(?:,\s*(\w+))?\s*\)', line)
        if not match:
            raise SyntaxError(f"输入语句语法错误: {line}")
        
        prompt, var_name = match.groups()
        
        if var_name is None:
            var_name = prompt
            prompt = f"请输入 {var_name} 的值:"
        else:
            prompt = prompt.strip('"\'')
        
        if var_name not in self.variables:
            raise NameError(f"未定义的变量: {var_name}")
        
        input_value = simpledialog.askstring("输入", prompt, parent=self.root_window)
        
        if input_value is None:
            input_value = ""
        
        var_type = type(self.variables[var_name]).__name__
        
        try:
            if var_type == 'int':
                self.variables[var_name] = int(input_value)
            elif var_type == 'float':
                self.variables[var_name] = float(input_value)
            elif var_type == 'bool':
                self.variables[var_name] = input_value.lower() == 'true'
            else:
                self.variables[var_name] = input_value
        except ValueError:
            raise ValueError(f"无法将输入值 '{input_value}' 转换为 {var_type}")
    
    def _handle_function_call(self, line: str):
        func_name = line.split('(', 1)[0]
        if func_name not in self.functions:
            raise NameError(f"未定义的函数: {func_name}")
        
        func_info = self.functions[func_name]
        
        args_str = line[len(func_name)+1:-1]
        args = [arg.strip() for arg in args_str.split(',')] if args_str else []
        
        if len(args) != len(func_info['params']):
            raise TypeError(f"函数 {func_name} 需要 {len(func_info['params'])} 个参数，但提供了 {len(args)} 个")
        
        evaluated_args = []
        for arg, (param_name, param_type) in zip(args, func_info['params']):
            value = self._evaluate_expression(arg)
            if param_type == 'int' and not isinstance(value, int):
                raise TypeError(f"参数 {param_name} 应为 {param_type} 类型")
            elif param_type == 'float' and not isinstance(value, float):
                raise TypeError(f"参数 {param_name} 应为 {param_type} 类型")
            elif param_type == 'bool' and not isinstance(value, bool):
                raise TypeError(f"参数 {param_name} 应为 {param_type} 类型")
            elif param_type == 'string' and not isinstance(value, str):
                raise TypeError(f"参数 {param_name} 应为 {param_type} 类型")
            
            evaluated_args.append(value)
        
        old_variables = self.variables.copy()
        old_line = self.current_line
        old_blocks = self.current_blocks.copy()
        
        self.variables = {}
        self.current_blocks = []
        
        for (param_name, _), value in zip(func_info['params'], evaluated_args):
            self.variables[param_name] = value
        
        self.current_line = func_info['start_line'] + 1
        end_line = func_info['end_line']
        
        while self.current_line < end_line:
            line = self.code_lines[self.current_line].strip()
            if line:
                self._process_line(line)
            self.current_line += 1
        
        return_value = self.return_value
        self.variables = old_variables
        self.current_line = old_line
        self.current_blocks = old_blocks
        self.return_value = None
        
        if func_info['return_type'] != 'void' and return_value is None:
            raise ValueError(f"函数 {func_name} 应返回 {func_info['return_type']} 类型的值")
        
        return return_value
    
    def _handle_assignment(self, line: str):
        var_name, expr = line.split('=', 1)
        var_name = var_name.strip()
        
        if var_name not in self.variables:
            raise NameError(f"未定义的变量: {var_name}")
        
        value = self._evaluate_expression(expr.strip())
        self.variables[var_name] = value
    
    def _handle_block_start(self):
        pass
    
    def _handle_block_end(self):
        if not self.current_blocks:
            raise SyntaxError("意外的块结束标记 ?>")
        
        current_block = self.current_blocks[-1]
        
        if current_block['type'] == 'loop':
            current_block['current_iteration'] += 1
            if current_block['current_iteration'] <= current_block['iterations']:
                self.current_line = current_block['start_line']
            else:
                self.current_blocks.pop()
        elif current_block['type'] == 'while':
            condition_met = self._evaluate_condition(current_block['condition'])
            if condition_met:
                self.current_line = current_block['start_line']
            else:
                self.current_blocks.pop()
        elif current_block['type'] == 'if':
            self.current_blocks.pop()
        else:
            raise SyntaxError("未知的块类型")
    
    def _handle_window_declaration(self, line: str):
        match = re.match(r'var Window (\w+)\s*=\s*Window\("(.+)",\s*(\d+),\s*(\d+)\)', line)
        if not match:
            raise SyntaxError("窗口声明语法错误")
        
        var_name, title, width, height = match.groups()
        self.root_window.after(0, lambda: self._create_window(var_name, title, int(width), int(height)))
    
    def _create_window(self, var_name: str, title: str, width: int, height: int):
        self.gui_objects[var_name] = GUIWindow(self.root_window, title, width, height)
        self.variables[var_name] = self.gui_objects[var_name]
    
    def _handle_button_declaration(self, line: str):
        match = re.match(r'var Button (\w+)\s*=\s*Button\("(.+)",\s*(\d+),\s*(\d+)\)', line)
        if not match:
            raise SyntaxError("按钮声明语法错误")
        
        var_name, text, x, y = match.groups()
        self.gui_objects[var_name] = GUIButton(text, int(x), int(y))
        self.variables[var_name] = self.gui_objects[var_name]
    
    def _handle_label_declaration(self, line: str):
        match = re.match(r'var Label (\w+)\s*=\s*Label\("(.+)",\s*(\d+),\s*(\d+)\)', line)
        if not match:
            raise SyntaxError("标签声明语法错误")
        
        var_name, text, x, y = match.groups()
        self.gui_objects[var_name] = GUILabel(text, int(x), int(y))
        self.variables[var_name] = self.gui_objects[var_name]
    
    def _find_block_end(self, start_line: int) -> int:
        block_level = 1
        current_line = start_line + 1
        
        while current_line < len(self.code_lines):
            line = self.code_lines[current_line].strip()
            if line == '<?':
                block_level += 1
            elif line == '?>':
                block_level -= 1
                if block_level == 0:
                    return current_line
            current_line += 1
        
        raise SyntaxError("未找到匹配的块结束标记 ?>")
    
    def _parse_value(self, value_str: str, target_type: str) -> Any:
        try:
            if target_type == 'int':
                return int(value_str)
            elif target_type == 'float':
                return float(value_str)
            elif target_type == 'bool':
                return value_str.lower() == 'true'
            elif target_type == 'string':
                if (value_str.startswith('"') and value_str.endswith('"')) or \
                   (value_str.startswith("'") and value_str.endswith("'")):
                    return value_str[1:-1]
                return value_str
            else:
                raise ValueError(f"不支持的类型: {target_type}")
        except ValueError as e:
            raise ValueError(f"无法将 '{value_str}' 转换为 {target_type}: {e}")
    
    def _evaluate_expression(self, expr: str) -> Any:
        if expr in self.variables:
            return self.variables[expr]
        
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]
        
        if expr.lower() == 'true':
            return True
        if expr.lower() == 'false':
            return False
        
        if expr.isdigit():
            return int(expr)
        if '.' in expr and all(p.isdigit() for p in expr.split('.', 1)):
            return float(expr)
        
        try:
            for var_name in self.variables:
                if var_name in expr:
                    expr = expr.replace(var_name, str(self.variables[var_name]))
            
            return eval(expr, {'__builtins__': None}, {})
        except:
            raise ValueError(f"无法评估表达式: {expr}")
    
    def _evaluate_condition(self, condition: str) -> bool:
        condition = condition.strip()
        
        if '(' in condition and ')' in condition:
            start = condition.find('(')
            end = condition.rfind(')')
            sub_condition = condition[start+1:end]
            sub_result = self._evaluate_condition(sub_condition)
            condition = condition[:start] + str(sub_result) + condition[end+1:]
        
        if '!' in condition:
            parts = condition.split('!', 1)
            right = self._evaluate_condition(parts[1])
            return not right
        
        if '|' in condition:
            parts = condition.split('|')
            left = self._evaluate_condition(parts[0])
            right = self._evaluate_condition('|'.join(parts[1:]))
            return left and right
        
        if '/' in condition:
            parts = condition.split('/')
            left = self._evaluate_condition(parts[0])
            right = self._evaluate_condition('/'.join(parts[1:]))
            return left or right
        
        for op in ['==', '!=', '>=', '<=', '>', '<']:
            if op in condition:
                left, right = condition.split(op, 1)
                left_val = self._evaluate_expression(left.strip())
                right_val = self._evaluate_expression(right.strip())
                
                if op == '==':
                    return left_val == right_val
                elif op == '!=':
                    return left_val != right_val
                elif op == '>=':
                    return left_val >= right_val
                elif op == '<=':
                    return left_val <= right_val
                elif op == '>':
                    return left_val > right_val
                elif op == '<':
                    return left_val < right_val
        
        return bool(self._evaluate_expression(condition))

class GUIWindow:
    def __init__(self, parent, title: str, width: int, height: int):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry(f"{width}x{height}")
        self.widgets = []
    
    def add(self, widget):
        if hasattr(widget, 'create'):
            widget.create(self.window)
            self.widgets.append(widget)
    
    def show(self):
        self.window.mainloop()

class GUIButton:
    def __init__(self, text: str, x: int, y: int):
        self.text = text
        self.x = x
        self.y = y
        self.callback = None
        self.tk_button = None
    
    def create(self, parent):
        self.tk_button = ttk.Button(parent, text=self.text)
        self.tk_button.place(x=self.x, y=self.y)
        if self.callback:
            self.tk_button.config(command=self.callback)
    
    def onClick(self, func: Callable):
        self.callback = func

class GUILabel:
    def __init__(self, text: str, x: int, y: int):
        self.text = text
        self.x = x
        self.y = y
        self.tk_label = None
    
    def create(self, parent):
        self.tk_label = ttk.Label(parent, text=self.text)
        self.tk_label.place(x=self.x, y=self.y)
    
    def setText(self, new_text: str):
        self.text = new_text
        if self.tk_label:
            self.tk_label.config(text=new_text)

if __name__ == "__main__":
    root = tk.Tk()
    ide = BiangIDE(root)
    root.mainloop()
