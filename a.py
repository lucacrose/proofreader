import os
import pathspec

# --- Configuration ---
OUTPUT_FILE = "repo_flattened.txt"
# Optional: Add extra ignores here that might not be in your .gitignore
EXTRA_IGNORES = {'.git', OUTPUT_FILE}

def get_gitignore_spec(root_dir):
    gitignore_path = os.path.join(root_dir, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', f.readlines())
        return spec
    return None

def flatten_repo(root_dir):
    spec = get_gitignore_spec(root_dir)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # Calculate relative path from root for matching
            rel_root = os.path.relpath(root, root_dir)
            if rel_root == ".":
                rel_root = ""

            # 1. Filter Directories
            dirs[:] = [d for d in dirs if d not in EXTRA_IGNORES]
            if spec:
                dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(rel_root, d))]

            # 2. Filter and Write Files
            for file in files:
                rel_file_path = os.path.join(rel_root, file)
                
                # Skip if in EXTRA_IGNORES or matches .gitignore
                if file in EXTRA_IGNORES:
                    continue
                if spec and spec.match_file(rel_file_path):
                    continue

                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()
                        outfile.write(f"--- FILE: {rel_file_path} ---\n")
                        outfile.write(content)
                        outfile.write(f"\n--- END OF {rel_file_path} ---\n\n")
                    print(f"Included: {rel_file_path}")
                except Exception as e:
                    print(f"Skipped {rel_file_path} (Error: {e})")

if __name__ == "__main__":
    path = os.getcwd()
    flatten_repo(path)
    print(f"\nSuccess! Repository flattened to {OUTPUT_FILE}")