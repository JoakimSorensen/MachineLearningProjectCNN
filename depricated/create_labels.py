import os
import shutil

train_directory = 'MathExprJpeg/extracted_images/'
val_dir = 'MathExprJpeg/test_directory/'

# make testing parent directory
os.makedirs(val_dir)

# make the labels txt files in respective directory
train_labels = open(train_directory + "labels.txt", 'w+')
val_labels = open(val_dir + 'labels.txt', 'w+')

labels = []
for(dirpath, dirnames, filenames) in os.walk(train_directory):
	labels.extend(dirnames)

for label in labels:
	os.makedirs(val_dir + label)
	train_labels.write(label + '\n')
	val_labels.write(label + '\n')

k = 0
for(dirpath, dirnames, filenames) in os.walk(train_directory):
	# take 10% to be tetsting data
	if not dirpath == train_directory:
		for i in range(len(filenames)//10):
			print(len(filenames))
			print(dirpath)
			shutil.copy(dirpath + '/' + filenames[i], val_dir + labels[k])
		k += 1
