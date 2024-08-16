import subprocess
import json
import os

CONFIG_FILE = "image_shuffling/configs.json"

def load_configs():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return []

def save_configs(configs):
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=4)

def build_command(config):
    # Check if shuffler.py is in the current directory
    shuffler_path = "shuffler.py"
    if not os.path.exists(shuffler_path):
        shuffler_path = "image_shuffling/shuffler.py"
    
    command = ["python3", shuffler_path]
    command.append(f"--input_dir {config['input_dir']}")
    command.append(f"--output_dir {config['output_dir']}")
    command.append(f"-r {config['r']}")
    command.append(f"-c {config['c']}")
    if config.get("save"):
        command.append("--save")
    command.append(f"-gr {config['gr']}")
    command.append(f"-gc {config['gc']}")
    if config.get("l"):
        command.append("-l")
    command.append(f"-lt {config['lt']}")
    return " ".join(command)

def run_command(command):
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

def create_custom_config():
    name = input("Enter a name for the configuration: ")
    input_dir = input("Enter input directory: ")
    output_dir = input("Enter output directory: ")
    r = int(input("Enter number of rows: "))
    c = int(input("Enter number of columns: "))
    save = input("Save output? (yes/no): ").lower() == "yes"
    gr = int(input("Enter grid rows: "))
    gc = int(input("Enter grid columns: "))
    l = input("Enable logging? (yes/no): ").lower() == "yes"
    lt = int(input("Enter logging threshold: "))

    return {
        "name": name,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "r": r,
        "c": c,
        "save": save,
        "gr": gr,
        "gc": gc,
        "l": l,
        "lt": lt
    }

def main():
    configs = load_configs()

    while True:
        print("\nAvailable configurations:")
        for i, config in enumerate(configs):
            print(f"{i + 1}. {config['name']}")

        print(f"{len(configs) + 1}. Create a new configuration")
        print(f"{len(configs) + 2}. Exit")

        choice = int(input("Choose an option: "))

        if choice == len(configs) + 1:
            new_config = create_custom_config()
            configs.append(new_config)
            save_choice = input("Save this configuration? (yes/no): ").lower()
            if save_choice == "yes":
                save_configs(configs)
            run_command(build_command(new_config))
            break
        elif choice == len(configs) + 2:
            break
        elif 1 <= choice <= len(configs):
            run_command(build_command(configs[choice - 1]))
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()