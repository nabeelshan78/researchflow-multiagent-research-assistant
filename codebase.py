import os

# Configuration: Directories and file extensions to ignore to keep the output clean
IGNORE_DIRS = {
    '.git', 'venv', 'env', '__pycache__', 'node_modules',
    '.idea', '.vscode', 'build', 'dist', '.pytest_cache', '.next'
}

IGNORE_EXTS = {
    # Compiled/Binary
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.class',
    # Media/Images
    '.jpg', '.jpeg', '.png', '.gif', '.ico', '.svg', '.webp', '.mp4', '.mp3',
    # Archives/Databases
    '.zip', '.tar', '.gz', '.rar', '.db', '.sqlite', '.sqlite3', '.pdf',
    # Lock files
    'package-lock.json', 'yarn.lock', 'poetry.lock'
}

OUTPUT_FILE = 'entire_codebase.txt'

def generate_codebase_txt(root_dir='.'):
    print("Scanning directory and compiling codebase...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # Modify dirs in-place to skip ignored directories entirely
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]

            for file in files:
                # Skip hidden files or files with ignored extensions
                if file.startswith('.') or any(file.endswith(ext) for ext in IGNORE_EXTS):
                    continue

                # Skip the output file itself to prevent self-replication
                if file == OUTPUT_FILE:
                    continue

                filepath = os.path.join(root, file)
                
                try:
                    # Attempt to read the file as text
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                    # Write clear visual separators and the file path
                    outfile.write(f"{'=' * 80}\n")
                    outfile.write(f"File: {filepath}\n")
                    outfile.write(f"{'=' * 80}\n\n")
                    outfile.write(content)
                    outfile.write("\n\n\n")
                    
                except UnicodeDecodeError:
                    # Silently skip binary files that bypassed the extension filter
                    pass
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    print(f"Done! All code has been saved to '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    generate_codebase_txt()