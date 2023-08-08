import os

def create_file_dict(directory, extensions):
    file_dict = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_dict[file] = os.path.join(root, file)
    return file_dict

def main():
    input_directory = "output/processed_imagery/area"
    compared_directories = ["output/polyline_images", "output/processed_imagery/point"]
    file_extensions = [".txt", ".jpg", ".png"]  # Add more extensions if needed

    input_files = create_file_dict(input_directory, file_extensions)

    for compared_directory in compared_directories:
        compared_files = create_file_dict(compared_directory, file_extensions)
        for filename in list(input_files.keys()):
            if filename not in compared_files:
                del input_files[filename]
                print(f"Removed {filename} from input directory.")

        for filename in list(compared_files.keys()):
            if filename not in input_files:
                file_path = compared_files[filename]
                os.remove(file_path)
                print(f"Removed {filename} from compared directory {compared_directory}.")

    for filename in list(input_files.keys()):
        for compared_directory in compared_directories:
            compared_files = create_file_dict(compared_directory, file_extensions)
            if filename not in compared_files:
                file_path = input_files[filename]
                os.remove(file_path)
                del input_files[filename]
                print(f"Removed {filename} from input directory.")

if __name__ == "__main__":
    main()

print("done.")