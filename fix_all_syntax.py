#!/usr/bin/env python3
"""
Fix ALL Python syntax errors in main_simple.py by systematically correcting indentation
"""

# Read the file
with open('main_simple.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Track indentation levels
fixed_lines = []
for i, line in enumerate(lines):
    original_line = line
    
    # Skip empty lines
    if not line.strip():
        fixed_lines.append(line)
        continue
    
    # Pattern 1: Fix import statements that are wrongly indented
    if line.strip().startswith('from utils.') or line.strip().startswith('import '):
        if 'try:' in ''.join(lines[max(0, i-5):i]):
            # This import is inside a try block, should have 4 spaces
            fixed_lines.append('    ' + line.strip() + '\n')
        else:
            # This import is at module level, should have 0 spaces
            fixed_lines.append(line.strip() + '\n')
        continue
    
    # Pattern 2: Fix except blocks
    if 'except Exception as e:' in line:
        # Find the matching try block
        try_indent = None
        for j in range(i-1, max(0, i-20), -1):
            if 'try:' in lines[j]:
                try_indent = len(lines[j]) - len(lines[j].lstrip())
                break
        
        if try_indent is not None:
            # Except should have same indentation as try
            fixed_lines.append(' ' * try_indent + 'except Exception as e:\n')
        else:
            # Default to no indentation
            fixed_lines.append('except Exception as e:\n')
        continue
    
    # Pattern 3: Fix lines following except blocks
    if i > 0 and 'except Exception as e:' in fixed_lines[i-1]:
        except_indent = len(fixed_lines[i-1]) - len(fixed_lines[i-1].lstrip())
        # Content inside except should be indented 4 more spaces
        if line.strip():
            fixed_lines.append(' ' * (except_indent + 4) + line.strip() + '\n')
        else:
            fixed_lines.append(line)
        continue
    
    # Pattern 4: Fix general indentation issues
    stripped = line.strip()
    if stripped:
        # Count leading spaces
        leading_spaces = len(line) - len(line.lstrip())
        
        # Common fixes for known patterns
        if stripped.startswith('logger.') or stripped.startswith('self.') or stripped.startswith('return ') or stripped.startswith('raise'):
            # These are usually inside methods or except blocks
            # Look for context
            context = ''.join(lines[max(0, i-10):i])
            if 'def ' in context and 'except' not in context:
                # Inside a method, should have 8 spaces
                fixed_lines.append('        ' + stripped + '\n')
            elif 'except' in lines[i-1] if i > 0 else False:
                # Inside an except block, figure out indentation from except
                prev_except_indent = len(lines[i-1]) - len(lines[i-1].lstrip()) if 'except' in lines[i-1] else 0
                fixed_lines.append(' ' * (prev_except_indent + 4) + stripped + '\n')
            else:
                # Keep original
                fixed_lines.append(line)
        else:
            # Keep original
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write the corrected file
with open('main_simple_corrected.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Created main_simple_corrected.py")

# Test syntax
try:
    with open('main_simple_corrected.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'main_simple_corrected.py', 'exec')
    print("✅ Syntax is now valid!")
except SyntaxError as e:
    print(f"❌ Still has error at line {e.lineno}: {e.msg}")
    print(f"Line content: {e.text.strip() if e.text else 'unknown'}")