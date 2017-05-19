import numpy as np
import cv2
import sys
import operator
from six.moves import cPickle as pickle


def main():

	target_image_size = 64
	num_channels = 3

	root_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/McGill/McGillFaces_public_v2/'

	images_to_pickle = []
	labels_to_pickle = np.zeros((6484, 1))


	# used to keep track of total images
	counter = 0


	for person in range (1, 61):

		# image files have person included in format 01, 02, 03...51, 52. this adds the zero to numbers lower than 10 (single digits)
		if person < 10:
			person_string = '0' + str(person)
		else: person_string = str(person)

		# some of the images have _0_ in the name, some have _1_
		# this was quicker than thinking of some clever way to work this out, there was no correlation
		if person == 18 or person == 21 or person == 23 or person == 24 or person ==26 or person == 27 or person == 35 or person == 40 or person == 45 or person == 49 or person == 52 or person ==57 or person == 60:
			additional_num = '0'
		else: additional_num ='1'


		# this first loop goes through each folder and checks atleast one image is present in that folder (images do not all start at 001)
		for image in range (1, 301):

			# image files have image number included in format 001, 002.. 021, 022... 101, 102. this adds two 0s to single digits, and 1 to double
			if image < 100:
				image_string = '0' + str(image)
				if image < 10:
					image_string = '0' + image_string
			else: image_string = str(image)

			# read the current image
			image_path = root_directory + 'Final/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
			image = cv2.imread(image_path)

			# not all images are present, so if this file read returned nothing, skip step
			if image is not None:

				print('person {0}'.format(person_string))

				# if an image was present, load labels from .txt file
				label_path = root_directory + person_string + '/pose/labels.txt'
				labels = np.loadtxt(label_path)

				# also load the corresponding tag (says which person and photo this labels belongs to)
				tags_path = root_directory + person_string + '/pose/tags.txt'
				tags =  np.genfromtxt(tags_path, dtype='str')

				# create dictionary object containing tag and label, order them by tag
				dictionary = dict(zip(tags, labels))
				dictionary = sorted(dictionary.items(), key=operator.itemgetter(0))

				# now cycle all the images in this folder again
				for image in range (1, 301):
					# image files have image number included in format 001, 002.. 021, 022... 101, 102. this adds two 0s to single digits, and 1 to double
					if image < 100:
						image_string = '0' + str(image)
						if image < 10:
							image_string = '0' + image_string
					else: image_string = str(image)

					image_path = root_directory + 'Final/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
					img = cv2.imread(image_path)
					
					# if image exists
					if img is not None:
						
						# add image and corresponding label to new arrays that will be pickled at the end
						key = person_string + '_' + additional_num + '_' + image_string
						for i in range(0, len(dictionary)):
							if key in dictionary[i]:

								images_to_pickle.append(img)
								labels_to_pickle[counter] = dictionary[i][1]

						counter += 1
				break


	# convert images to format that can be pickled and read into experiment script easilt
	training_array = np.asarray(images_to_pickle)
	training_array = training_array.astype(dtype=np.float32)
	training_array /= 127
	training_array = np.reshape(training_array, (-1, 64*64*3))

	# dump all images and labels in pickle 
	pickle_file_name = 'McGill.pickle'
	file = open(pickle_file_name, 'wb')
	save = {'data': training_array,
			'labels': labels_to_pickle}
	pickle.dump(save, file)
	file.close()
	
	
	
	






		








			




if __name__ == "__main__":
    main()