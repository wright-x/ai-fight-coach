#!/usr/bin/env python3
"""
Fix all Python syntax errors in main_simple.py by correcting indentation
"""

import re

# Read the file
with open('main_simple.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Fix patterns
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Pattern 1: Fix orphaned except blocks (8 spaces should be 4 spaces if following a method)
    if '        except Exception as e:' in line:
        # Check if we're inside a function/method (look for recent 'def' or 'try')
        context = ''.join(lines[max(0, i-10):i]).lower()
        if 'def ' in context and 'try:' in context:
            # This should be method-level except (8 spaces)
            fixed_lines.append('        except Exception as e:\n')
        else:
            # This should be module-level except (0 spaces)
            fixed_lines.append('except Exception as e:\n')
    
    # Pattern 2: Fix any except at wrong indentation level
    elif line.strip() == 'except Exception as e:' and not line.startswith('    ') and not line.startswith('        '):
        # Find the matching try block indentation
        context_lines = lines[max(0, i-15):i]
        try_indent = None
        for ctx_line in reversed(context_lines):
            if 'try:' in ctx_line:
                try_indent = len(ctx_line) - len(ctx_line.lstrip())
                break
        
        if try_indent is not None:
            fixed_lines.append(' ' * try_indent + 'except Exception as e:\n')
        else:
            fixed_lines.append('except Exception as e:\n')
    
    # Pattern 3: Fix lines that follow except blocks
    elif i > 0 and 'except Exception as e:' in fixed_lines[i-1]:
        if line.strip() and not line.startswith('    ') and not line.startswith('        '):
            # This line should be indented relative to the except
            except_indent = len(fixed_lines[i-1]) - len(fixed_lines[i-1].lstrip())
            fixed_lines.append(' ' * (except_indent + 4) + line.strip() + '\n')
        else:
            fixed_lines.append(line)
    
    else:
        fixed_lines.append(line)
    
    i += 1

# Write the fixed file
with open('main_simple_fixed.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Created main_simple_fixed.py with corrected syntax")

# Test the syntax
try:
    with open('main_simple_fixed.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'main_simple_fixed.py', 'exec')
    print("✅ Fixed file has valid Python syntax!")
except SyntaxError as e:
    print(f"❌ Still has syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")