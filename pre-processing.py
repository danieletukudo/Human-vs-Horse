import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


train_horse_dir = os.path.join("horse-or-human/train/horses")

train_human_dir = os.path.join("horse-or-human/train/humans")
validation_dation_horse_dir = os.path.join('horse-or-human/horse-or-human/validation/horses')
validation_dation_human_dir = os.path.join('horse-or-human/horse-or-human/validation/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

val_horse_names = os.listdir(validation_dation_horse_dir)
print(val_horse_names[:10])

val_human_names = os.listdir(validation_dation_human_dir)
print(val_human_names[:10])


print("total train horse image", len(os.listdir(train_horse_dir)))
print("total train human image", len(os.listdir(train_human_dir)))

print("total val horse image", len(os.listdir(validation_dation_horse_dir)))
print("total val human image", len(os.listdir(validation_dation_human_dir)))

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8
next_horse_pix = [os.path.join(validation_dation_horse_dir,fname) for fname in val_horse_names [pic_index - 8:pic_index]]
next_human_pix = [os.path.join(validation_dation_human_dir,fname) for fname in val_human_names [pic_index -8:pic_index]]

for i , img_path in enumerate (next_horse_pix + next_human_pix):

    sp = plt.subplot(nrows,ncols, i+1)
    sp.axis('off')

    img = mpimg.imread(img_path)
    plt.imshow (img)
plt.show()


# Set u