"""
Test runner script for the TVC-AI project.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run test suite with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    
    # Configure test selection
    if test_type == "unit":
        cmd.extend([str(test_dir / "test_environment.py"), str(test_dir / "test_agent.py")])
    elif test_type == "integration":
        cmd.extend([str(test_dir / "test_integration.py"), "-m", "integration"])
    elif test_type == "benchmark":
        cmd.extend([str(test_dir / "benchmark.py")])
    elif test_type == "fast":
        cmd.extend([str(test_dir), "-m", "not slow"])
    else:  # all
        cmd.append(str(test_dir))
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage reporting
    if coverage:
        cmd.extend([
            "--cov=agent",
            "--cov=env", 
            "--cov=scripts",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Additional useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Error on unknown markers
        "-ra"  # Show summary of all test results
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run TVC-AI test suite")
    
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "benchmark", "fast"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running"
    )
    
    args = parser.parse_args()
    
    # Install test dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "pytest", "pytest-cov", "psutil"
        ])
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
