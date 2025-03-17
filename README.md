# VasthraAI Implementation Guildlines![Vasthra WHITE](https://github.com/user-attachments/assets/e1ceb98c-de8a-43fb-9eb9-b66e95552177)

## Using the preprocessing scripts

### scrape.py

- This script can be used to scrape images out of complex datasets. It scans for image related filetypes and copies them into a target directory.
- Run it using a simple python command and change the ouput and input directories from inside the script.

```bash
python scrape.py
```

### resize_color.py

- This script is will crop images into 512x512 images that can be used by the GAN. It also converts greyscale images into RGB. To use, run using the following arguments.

```bash
python resize_color.py --input_folder "C:\Input\Folder\Path" --output_folder "C:\Output\Folder\Path"
```

### resize_color.py

## Generating an image from a sketch

- Navigate to the desired model directory (GEN_x) as per your preference
- Run the generate_image.py followed by a command-line argument for the path to your sketch as follows.

```bash
python generate_image.py --sketch_path "C:\Users\Bimsara\Desktop\sample\sketch.png"
```
