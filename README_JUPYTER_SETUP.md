# Jupyter Notebook Extensions Setup

## Installed Extensions

This project has been configured with the following Jupyter extensions to enhance your notebook editing experience:

### 1. **nbdime** - Notebook Diff and Merge
- **Purpose**: Better handling of Jupyter notebook version control and merge conflicts
- **Features**:
  - Visual diff tool for notebooks
  - Smart merge conflict resolution
  - Git integration for `.ipynb` files
  
**Usage**:
```bash
# View diff between notebooks
nbdiff notebook1.ipynb notebook2.ipynb

# Merge notebooks with conflicts
nbmerge base.ipynb local.ipynb remote.ipynb

# Launch web-based diff viewer
nbdiff-web notebook1.ipynb notebook2.ipynb

# Launch web-based merge tool (useful for resolving conflicts)
nbmerge-web Load-Default_Prediction.ipynb
```

**Git Integration**: nbdime is now configured globally for git. When you encounter merge conflicts in `.ipynb` files, you can use:
```bash
git mergetool --tool=nbdime
```

### 2. **JupyterLab Git Extension**
- **Purpose**: Git integration directly in JupyterLab UI
- **Features**:
  - Visual git interface in JupyterLab
  - Commit, push, pull from the UI
  - View diffs and history
  
**Usage**: Look for the Git icon in the left sidebar of JupyterLab

### 3. **ipywidgets** - Interactive Widgets
- **Purpose**: Create interactive visualizations and controls
- **Features**:
  - Sliders, dropdowns, buttons
  - Interactive plots
  - Progress bars
  
**Usage**:
```python
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt

@interact(x=(0, 10, 0.1))
def plot_function(x=5):
    plt.plot([1, 2, 3], [x, x*2, x*3])
    plt.show()
```

### 4. **Black** - Code Formatter
- **Purpose**: Automatic Python code formatting
- **Features**:
  - Consistent code style
  - PEP 8 compliant
  - Jupyter notebook support
  
**Usage**:
```bash
# Format a Python file
black your_script.py

# Format code in a notebook
black Load-Default_Prediction.ipynb

# Check what would be formatted (dry run)
black --check Load-Default_Prediction.ipynb
```

### 5. **isort** - Import Sorter
- **Purpose**: Automatically sort and organize imports
- **Features**:
  - Groups imports by type
  - Alphabetical sorting
  - Compatible with Black
  
**Usage**:
```bash
# Sort imports in a file
isort your_script.py

# Sort imports in notebook
isort Load-Default_Prediction.ipynb
```

## Quick Start

### Launch JupyterLab
```bash
jupyter lab
```

### Resolve Merge Conflicts in Notebooks

Your `Load-Default_Prediction.ipynb` currently has merge conflict markers. To resolve:

**Option 1: Using nbdime web tool (Recommended)**
```bash
nbmerge-web Load-Default_Prediction.ipynb
```

**Option 2: Manual resolution**
1. Open the notebook in a text editor
2. Look for conflict markers: `<<<<<<< HEAD`, `=======`, `>>>>>>> branch-name`
3. Remove the conflict markers and keep the desired code
4. Save the file

**Option 3: Using nbdime with git**
```bash
git mergetool --tool=nbdime Load-Default_Prediction.ipynb
```

## Additional Tips

### Prevent Merge Conflicts
To minimize merge conflicts in notebooks:

1. **Clear outputs before committing**:
   ```bash
   jupyter nbconvert --clear-output --inplace Load-Default_Prediction.ipynb
   ```

2. **Use nbstripout** (optional):
   ```bash
   pip install nbstripout
   nbstripout --install  # Automatically strips output on commit
   ```

### Best Practices for Notebook Version Control

1. **Commit frequently** with small, focused changes
2. **Clear cell outputs** before committing (reduces diff noise)
3. **Use descriptive commit messages**
4. **Pull before starting work** to avoid conflicts
5. **Use nbdime** for reviewing notebook changes

## Installed Packages

All dependencies are tracked in `requirements.txt`. To install them:

```bash
pip install -r requirements.txt
```

## Project Structure

```
good-loan-approvals/
├── Load-Default_Prediction.ipynb  # Main analysis notebook
├── Prep4Test.ipynb                # Test preparation notebook
├── pipeline.ipynb                 # Pipeline notebook
├── data/                          # Data directory
├── Streamlit/                     # Streamlit app
├── requirements.txt               # Python dependencies
└── README_JUPYTER_SETUP.md        # This file
```

## Troubleshooting

### Extension not showing in JupyterLab
```bash
jupyter labextension list  # Check installed extensions
jupyter lab build          # Rebuild JupyterLab
```

### nbdime not working with git
```bash
nbdime config-git --enable --global
```

### Kernel issues
```bash
jupyter kernelspec list    # List available kernels
python -m ipykernel install --user --name=myenv  # Install kernel
```

## Resources

- [nbdime Documentation](https://nbdime.readthedocs.io/)
- [JupyterLab Git Extension](https://github.com/jupyterlab/jupyterlab-git)
- [ipywidgets Documentation](https://ipywidgets.readthedocs.io/)
- [Black Documentation](https://black.readthedocs.io/)
