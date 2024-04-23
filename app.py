import pyperclip

# Define program titles for DAV and ML subjects
dav_programs = [
    "Introduction to Python",
    "Data Structures in Python",
    "Object-Oriented Programming",
    "Web Development Basics",
    "Database Management Systems",
    "Machine Learning Basics",
    "Deep Learning Fundamentals",
    "Natural Language Processing",
    "Computer Vision",
    "Big Data Analytics"
]

ml_programs = [
    "Linear Regression",
    "Logistic Regression",
    "Decision Trees",
    "Random Forests",
    "Support Vector Machines",
    "K-Nearest Neighbors",
    "Clustering Algorithms",
    "Dimensionality Reduction",
    "Neural Networks",
    "Reinforcement Learning"
]

def display_programs(subject):
    programs = dav_programs if subject.lower() == 'dav' else ml_programs
    print(f"Total Programs for {subject.upper()}:")
    for idx, program in enumerate(programs, start=1):
        print(f"{idx}. {program}")

def get_code(subject, program_number):
    programs = dav_programs if subject.lower() == 'dav' else ml_programs
    if 1 <= program_number <= len(programs):
        program_title = programs[program_number - 1]
        program_file_name = f"program{program_number}.py"
        try:
            with open(program_file_name, "r") as file:
                code = file.read()
                print(code)
                pyperclip.copy(code)
                print("Code copied to clipboard.")
        except FileNotFoundError:
            print("Code file not found.")
    else:
        print("Invalid program number.")

def main():
    subject = input("Select subject (DAV or ML): ")
    if subject.lower() not in ['dav', 'ml']:
        print("Invalid subject selection.")
        return
    display_programs(subject)
    program_number = int(input("Enter the number of the program you want to see the code for (1-10): "))
    get_code(subject, program_number)

if __name__ == "__main__":
    main()
