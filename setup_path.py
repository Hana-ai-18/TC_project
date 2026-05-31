"""
setup_path.py — Add to Kaggle notebook to set up Python path correctly.

Usage in notebook:
    import sys
    sys.path.insert(0, '/kaggle/working/SRC_Track')
    exec(open('/kaggle/working/SRC_Track/setup_path.py').read())
    # OR just:
    sys.path.insert(0, '/kaggle/working/SRC_Track/src_track_new')
"""
import os
import sys

# This file lives in src_track_new/
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Also add parent in case called from above
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

print(f"[setup_path] Added to sys.path:")
print(f"  {_HERE}")
