import os
import subprocess

env_file = ".env"

if not os.path.exists(env_file):
    # Create a new .env file
    open(env_file, "w").close()
else:
    # .env file already exists, open it
    if os.name == "nt":  # Windows
        subprocess.run(["notepad.exe", env_file])
    elif os.name == "posix":  # Unix-based (macOS, Linux)
        if subprocess.run(["which", "open"]).returncode == 0:  # macOS
            subprocess.run(["open", "-a", "TextEdit", env_file])
        else:  # Linux
            subprocess.run(["xdg-open", env_file])