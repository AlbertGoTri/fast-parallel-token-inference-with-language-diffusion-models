#!/usr/bin/env python
"""
Validation script for Mistral-based Promptfoo evaluation.
Checks all dependencies and configurations before running evaluation.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_command(cmd, name):
    """Check if a command exists in PATH."""
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {name}: {result.stdout.strip().split(chr(10))[0]}")
            return True
    except (FileNotFoundError, OSError):
        pass
    print(f"✗ {name}: NOT FOUND")
    return False

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}: installed")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    print(f"✗ {description}: NOT FOUND at {filepath}")
    return False

def check_llada_server(url="http://127.0.0.1:5000/health"):
    """Check if LLaDA server is running."""
    try:
        import urllib.request
        response = urllib.request.urlopen(url, timeout=5)
        print(f"✓ LLaDA server: running at http://127.0.0.1:5000")
        return True
    except Exception:
        print(f"✗ LLaDA server: NOT RUNNING (start with: python serve_llada.py)")
        return False

def main():
    print("=" * 60)
    print("LLaDA Mistral-based Promptfoo Validation")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # System commands
    print("\n[System Commands]")
    commands = [
        ("node", "Node.js"),
        ("npx", "NPX"),
        ("python", "Python"),
    ]
    for cmd, name in commands:
        checks_total += 1
        if check_command(cmd, name):
            checks_passed += 1
    
    # Python packages
    print("\n[Python Packages]")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]
    for pkg, import_name in packages:
        checks_total += 1
        if check_python_package(pkg, import_name):
            checks_passed += 1
    
    # File existence
    print("\n[Configuration Files]")
    script_dir = Path(__file__).parent
    
    files_to_check = [
        (script_dir / "mistral_judge_provider.py", "Mistral Judge Provider"),
        (script_dir / "promptfooconfig_mistral.yaml", "Mistral Config"),
        (script_dir / "llada_api_provider.py", "LLaDA API Provider"),
        (script_dir / "run_evaluation.ps1", "Runner Script"),
        (script_dir / "generate_report.py", "Report Generator"),
    ]
    
    for filepath, description in files_to_check:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Optional: check server status (don't fail if not running)
    print("\n[Runtime Services] (optional - start if needed)")
    print("○ Checking LLaDA server...")
    check_llada_server()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Validation: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n✓ All checks passed! Ready to run:")
        print("  .\evaluation\promptfoo\run_evaluation.ps1")
        return 0
    else:
        failed = checks_total - checks_passed
        print(f"\n✗ {failed} check(s) failed. See above for details.")
        print("\nInstall missing packages with:")
        print("  pip install torch transformers numpy tqdm")
        print("\nOr install Node.js from: https://nodejs.org/")
        return 1

if __name__ == "__main__":
    sys.exit(main())
