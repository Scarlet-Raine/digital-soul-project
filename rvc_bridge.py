import os
import sys
import json
import traceback
import subprocess
import time

def run_rvc_process(input_file, output_dir="audio/out"):
    """Run RVC on the input file using direct command-line call"""
    try:
        # Create output directories if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"out_{timestamp}.wav")
        
        # Define paths to model files
        pth_path = os.path.join("models", "2BJP.pth")
        index_path = os.path.join("models", "2BJP.index")
        
        # Use subprocess to directly call rvc_inf_cli with the right parameters
        cmd = [
            sys.executable, 
            os.path.join("rvc_cli", "rvc_inf_cli.py"),  # Use local path in rvc_cli folder
            "infer",
            "--input_path", input_file,
            "--output_path", output_path,
            "--pth_path", pth_path,
            "--index_path", index_path,
            "--pitch", "0",
            "--filter_radius", "3",
            "--protect", "0.4",
            "--hop_length", "128",
            "--f0_method", "rmvpe"
        ]
        
        # Run the command and wait for it to complete
        print(f"Running RVC: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            print(f"RVC succeeded, output: {output_path}")
            return True, output_path
        else:
            print(f"RVC failed: {process.stderr}")
            return False, process.stderr
    
    except Exception as e:
        print(f"Error processing RVC: {str(e)}")
        traceback.print_exc()
        return False, str(e)

# Server mode - run when this script is executed directly
if __name__ == "__main__":
    print("RVC Server Ready", file=sys.stderr)
    sys.stderr.flush()
    
    while True:
        try:
            # Get input file from stdin
            input_file = input().strip()
            if not input_file:
                continue
                
            # Process the file
            success, result = run_rvc_process(input_file)
            
            # Report result
            if success:
                print(f"Success: {result}")
            else:
                print(f"Error: {result}")
                
            sys.stdout.flush()
            
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.stdout.flush()