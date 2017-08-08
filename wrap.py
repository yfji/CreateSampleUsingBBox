import os

root=os.getcwd()

# In[1]
"""
file_name='hard_samples.txt'

image_names=os.listdir(os.path.join(root,'train/hard'))

with open(file_name,'w') as f:
    for name in image_names:
        f.write(os.path.join(root,'train/hard', name)+'\n')
"""


# In[2]
label_file_name='label_ubuntu.txt'

pos_root=os.path.join(root, 'train/pot')
neg_root=os.path.join(root, 'train/nonpot')

pos_samples=os.listdir(pos_root)
neg_samples=os.listdir(neg_root)

with open(label_file_name, 'w') as f:
    for s in pos_samples:   
        f.write(os.path.join(pos_root, s)+' %d\n'%1)
    for s in neg_samples:
        f.write(os.path.join(neg_root, s)+' %d\n'%(-1))

print('done')