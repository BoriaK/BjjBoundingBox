import os
import re

# Correctly defined directory path using raw string
my_dir = r'C:\Users\BorisK\Documents\Python Projects\BjjBoundingBox\Dataset\Sub_Set_v01\train\bed'

# Regex pattern to match the specific path segment using raw string
# replace_what = r'(?<=<path>)(C:\\Users\\bkupcha\\OneDrive - Intel Corporation\\Documents\\PythonProjects\\Furnecher_Project_Pytorch\\Dataset\\Sub_Set_v01\\train\\tables)(?=\\img)'
replace_what = r'(?<=<path>)(C:\\Users\\BorisK\\Documents\\Python Projects\\BjjBoundingBox\\Dataset\\Sub_Set_v01\\train\\chairs)(?=\\img)'

# Replacement string using raw string
replace_with = r'C:\\Users\\BorisK\\Documents\\Python Projects\\BjjBoundingBox\\Dataset\\Sub_Set_v01\\train\\bed'

# Loop through all files in directory recursively
for root, directories, filenames in os.walk(my_dir):
    for filename in filenames:
        if filename.endswith('.xml'):
            file_path = os.path.join(root, filename)

            # Check if the file is a regular file
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                # Perform the regex replacement
                new_content = re.sub(replace_what, replace_with, content)

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8', errors='ignore') as file:
                    file.write(new_content)

print("Replacement complete.")
