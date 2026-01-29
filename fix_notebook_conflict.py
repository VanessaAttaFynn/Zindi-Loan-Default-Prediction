#!/usr/bin/env python3
"""
Script to fix merge conflicts in Jupyter notebooks
This script removes git merge conflict markers and fixes duplicate fields
"""

import json
import sys
import re

def fix_notebook_merge_conflict(notebook_path):
    """
    Fix merge conflicts in a Jupyter notebook by:
    1. Reading the file as text
    2. Removing git conflict markers
    3. Fixing duplicate execution_count fields
    4. Validating JSON structure
    """
    
    # Read the file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove git conflict markers
    lines = content.split('\n')
    cleaned_lines = []
    skip_until_end = False
    
    for line in lines:
        # Check for conflict markers
        if line.strip().startswith('<<<<<<< HEAD'):
            skip_until_end = False
            continue
        elif line.strip().startswith('======='):
            skip_until_end = True
            continue
        elif line.strip().startswith('>>>>>>>'):
            skip_until_end = False
            continue
        
        # Skip lines between ======= and >>>>>>>
        if not skip_until_end:
            cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Fix duplicate execution_count by using regex
    # Pattern: "outputs": [], followed by "execution_count": number
    pattern = r'("outputs"\s*:\s*\[\s*\]),\s*"execution_count"\s*:\s*(\d+)'
    replacement = r'"outputs": [], "execution_count": \2'
    
    # This won't work if execution_count appears twice in same cell
    # So we need to find and remove the first occurrence
    pattern2 = r'"execution_count"\s*:\s*\d+,\s*("id".*?"outputs"\s*:\s*\[\s*\]),\s*"execution_count"\s*:\s*(\d+)'
    replacement2 = r'\1, "execution_count": \2'
    
    cleaned_content = re.sub(pattern2, replacement2, cleaned_content)
    
    # Try to parse as JSON to validate
    try:
        notebook_data = json.loads(cleaned_content)
        
        # Additional cleanup: ensure each cell has execution_count only once
        if 'cells' in notebook_data:
            for cell in notebook_data['cells']:
                if cell.get('cell_type') == 'code':
                    # Ensure execution_count is at the end of the cell dict
                    if 'execution_count' in cell:
                        exec_count = cell.pop('execution_count')
                        cell['execution_count'] = exec_count
        
        # Write back with proper formatting
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully fixed merge conflicts in {notebook_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON after cleanup: {e}")
        print(f"  Saving cleaned content to {notebook_path}.cleaned for manual review")
        
        with open(f"{notebook_path}.cleaned", 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_notebook_conflict.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    success = fix_notebook_merge_conflict(notebook_path)
    sys.exit(0 if success else 1)
