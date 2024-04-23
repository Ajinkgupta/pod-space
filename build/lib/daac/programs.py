import pyperclip
import os

def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'program.py')
    with open(file_path, 'r') as file:
        code = file.read()
        pyperclip.copy(code)
        print("Code copied to clipboard!")

if __name__ == "__main__":
    main()
