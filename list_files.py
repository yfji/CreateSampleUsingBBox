import os

root=os.getcwd()

image_dir=os.path.join(root,'person_02', 'person')

image_files=os.listdir(image_dir)

with open(os.path.join(root, 'person_02.txt'), 'w') as f:
    for name in image_files:
        name=os.path.join(image_dir, name)
        f.write(name+'\n')

print('done')