import os
import exifread

from pathlib import Path  # import Path from pathlib module

directory = Path('H:/gedruckt')  # set directory path

for subdir in directory.iterdir():  
    for file in subdir.iterdir():
        if file.is_file():  # Check if it's a file
            ending = file.name.split('.')[1]
            with open(file, 'rb') as f:
                tags = exifread.process_file(f, details=True)
            f.close()
            iso = str(tags["EXIF ISOSpeedRatings"])
            f_num = str(tags["EXIF FNumber"])
            exp_time = str(tags["EXIF ExposureTime"]).replace("/", "-")
            name = f"ISO{iso}_F{f_num}_sh{exp_time}.{ending}"
            try:
                os.rename(file, subdir / name)
            except:
                print(file)
                print(name)
                print("Already existed")
