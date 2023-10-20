import os
import matplotlib.pyplot as plt
import cv2 as cv

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_text_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

def append_to_text_file(file_path, text, row_number):
    lines = read_text_file(file_path)
    
    if row_number < len(lines):
        lines[row_number] = lines[row_number].strip() + " " + text + "\n"

    write_text_file(file_path, lines)
    
def get_path(dir,file):
    dir_path = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(file):
                dir_path.append(file_path)
    return dir_path

def main():
    # Specify data paths
    dir1 = "Split_Dataset/train/annotations/"
    dir2 = "Split_Dataset/train/images/"
    annos = get_path(dir1, ".txt")
    imgs = get_path(dir2, ".jpg")
                
    for anno, img in zip(annos,imgs):
        print(f"img_path: {img}\nContents of {anno}:")
        lines = read_text_file(anno)
        for i, line in enumerate(lines):
            print(f"{i + 1}: {line.strip()}")
        image = cv.imread(img)
        plt.imshow(image)
        plt.title(f"{img}")
        plt.axis('off')
        plt.show(block=False)
        while True:
            try:
                row_number = input("Enter the row number to append to (n to NEXT / qq to QUIT): ")
                if row_number == "n":
                    print()
                    plt.close()
                    break
                elif row_number == "qq":
                    plt.close()
                    exit()
                    
                user_input = input("Enter text to append to the selected row: ")
                append_to_text_file(anno, user_input, int(row_number) - 1)
                lines = read_text_file(anno)
                print("\nAdded lines:")
                print(f"{anno}:")
                for i, line in enumerate(lines):
                    print(f"{i + 1}: {line.strip()}")
                print()
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid row number.")
                continue

if __name__ == '__main__':
    
    main()
