from pathlib import Path
import pandas as pd

def parse_m2(m2_file):
    """Parse m2 file into one row per error."""
    
    results = []
    current_source = None
    
    with open(m2_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("S "):
                current_source = line[2:]
            
            elif line.startswith("A "):
                parts = line.split("|||")
                
                # handle noop - sentence explicitly marked as correct
                if parts[1] == "noop":
                    results.append({
                        "source": current_source,
                        "start_tok": -1,
                        "end_tok": -1,
                        "err_type": "correct",
                        "correction": ""
                    })
                    continue
                
                # extract token positions
                span = parts[0].replace("A ", "").strip().split()
                start_tok = int(span[0])
                end_tok = int(span[1])
                
                error_type = parts[1]
                correction = parts[2]
                
                results.append({
                    "source": current_source,
                    "start_tok": start_tok,
                    "end_tok": end_tok,
                    "err_type": error_type,
                    "correction": correction
                })
    
    print(f"Parsed {len(results)} errors from M2 file")
    return pd.DataFrame(results)