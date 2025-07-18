#!/usr/bin/env python
import os
import sys
import subprocess
import argparse

def convert_path(windows_path):
    """Convert Windows path to Docker container path"""
    # Get the absolute path
    abs_path = os.path.abspath(windows_path)
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if the path is within the readT directory
    if "readT" in abs_path:
        # Extract the part of the path after "readT"
        parts = abs_path.split("readT")
        if len(parts) > 1:
            relative_path = parts[1].replace("\\", "/")
            # Ensure the path starts with a slash
            if not relative_path.startswith("/"):
                relative_path = "/" + relative_path
            return f"/app/readT{relative_path}"
    
    # If we can't determine the Docker path, return the original path
    print(f"Warning: Could not convert path {windows_path} to Docker container path.")
    return windows_path

def main():
    parser = argparse.ArgumentParser(description="Run text analysis in Docker container")
    parser.add_argument("input_path", help="Path to the input file or directory (Windows path)")
    parser.add_argument("--save-words", choices=["yes", "no", "ask"], default="ask",
                        help="Whether to save new words to database: yes, no, or ask (default)")
    parser.add_argument("--config", type=str, default="config.ini", 
                        help="Path to configuration file")
    parser.add_argument("--min-word-length", type=int, default=3,
                        help="Minimum length of words to include in analysis (default: 3)")
    
    args = parser.parse_args()
    
    # Convert Windows path to Docker container path
    docker_path = convert_path(args.input_path)
    
    # Build the Docker command
    cmd = [
        "docker", "exec", "-it", "text_analyzer", 
        "python", "src/main.py", 
        docker_path
    ]
    
    # Add optional arguments
    if args.save_words:
        cmd.extend(["--save-words", args.save_words])
    
    if args.config != "config.ini":
        cmd.extend(["--config", args.config])
    
    if args.min_word_length != 3:
        cmd.extend(["--min-word-length", str(args.min_word_length)])
    
    # Print the command being executed
    print(f"Running: {' '.join(cmd)}")
    
    # Execute the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    main()
