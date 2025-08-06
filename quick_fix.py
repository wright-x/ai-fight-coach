#!/usr/bin/env python3
"""
Quick fix for main_simple.py - fix ONLY the critical indentation errors
"""

import re

# Read file
with open('main_simple.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Fix the most critical patterns
# Pattern 1: Fix misaligned except blocks that should be at function level (4 spaces from start)
content = re.sub(r'(\n    .*?\n)        except Exception as e:', r'\1    except Exception as e:', content, flags=re.DOTALL)

# Pattern 2: Fix misaligned except blocks that should be at method level (8 spaces from start) 
content = re.sub(r'(\n        .*?\n)    except Exception as e:', r'\1        except Exception as e:', content, flags=re.DOTALL)

# Pattern 3: Fix orphaned except blocks (no matching indentation)
content = re.sub(r'\n        except Exception as e:\n        ([a-zA-Z])', r'\n    except Exception as e:\n        \1', content)

# Write fixed content
with open('main_simple.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied quick fixes to main_simple.py")

# Test syntax
import ast
try:
    ast.parse(content)
    print("✅ Syntax is now valid!")
except SyntaxError as e:
    print(f"❌ Still has error at line {e.lineno}: {e.msg}")
    if e.text:
        print(f"Line: {e.text.strip()}")