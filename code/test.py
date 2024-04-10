import os

# get every text file in the working directory
old_time = 0
newest_file_path = ""
for file in os.listdir():
    print(file)
    if file.endswith(".txt"):
        # get relative file path
        path = os.path.join(file)
        # get the creation time of the file
        ti_c = os.path.getctime(path)
        if ti_c > old_time:
            newest_file_path = path

current_folder = newest_file_path 
#print("currentfolder ", current_folder[:-4])
#print(current_folder[:-4])
x= current_folder[:-4].replace(" ", "_")
os.makedirs(x)