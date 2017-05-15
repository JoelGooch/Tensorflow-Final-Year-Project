import numpy as np
import scipy
import scipy.io as sio
import cv2
import sys
import mask_analysis
import operator
from six.moves import cPickle as pickle

np.set_printoptions(threshold=np.inf)

def main():

	mask_analyser = mask_analysis.BinaryMaskAnalyser()

	target_image_size = 64
	num_channels = 3

	root_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/McGill/McGillFaces_public_v2/'

	images_to_pickle = []
	labels_to_pickle = np.zeros((6484, 1))

	counter = 0


	for person in range (1, 61):

		if person < 10:
			person_string = '0' + str(person)
		else: person_string = str(person)

		if person == 18 or person == 21 or person == 23 or person == 24 or person ==26 or person == 27 or person == 35 or person == 40 or person == 45 or person == 49 or person == 52 or person ==57 or person == 60:
			additional_num = '0'
		else: additional_num ='1'


		for image in range (1, 301):

			if image < 100:
				image_string = '0' + str(image)
				if image < 10:
					image_string = '0' + image_string
			else: image_string = str(image)

			
			image_path = root_directory + 'Final/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
			#print(image_path)
			image = cv2.imread(image_path)

			if image is not None:
				#cv2.imshow('resized image', image)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()

				print('person {0}'.format(person_string))

				label_path = root_directory + person_string + '/pose/labels.txt'
				labels = np.loadtxt(label_path)
				#print('length of labels = {0}'.format(len(labels)))

				tags_path = root_directory + person_string + '/pose/tags.txt'
				tags =  np.genfromtxt(tags_path, dtype='str')
				#print('length of tags = {0}'.format(len(tags)))

				dictionary = dict(zip(tags, labels))
				dictionary = sorted(dictionary.items(), key=operator.itemgetter(0))
				#print('length of dict = {0}'.format(len(dictionary)))


				for image in range (1, 301):
					if image < 100:
						image_string = '0' + str(image)
						if image < 10:
							image_string = '0' + image_string
					else: image_string = str(image)

					image_path = root_directory + 'Final/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
					
					img = cv2.imread(image_path)
					
					if img is not None:
						
						key = person_string + '_' + additional_num + '_' + image_string
						for i in range(0, len(dictionary)):
							if key in dictionary[i]:

								images_to_pickle.append(img)
								labels_to_pickle[counter] = dictionary[i][1]

						counter += 1

				break


	print('images')
	print(len(images_to_pickle))
	cv2.imshow('resized image', images_to_pickle[0])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	training_array = np.asarray(images_to_pickle)
	training_array = training_array.astype(dtype=np.float32)
	training_array /= 127
	training_array = np.reshape(training_array, (-1, 64*64*3))

	print(training_array.shape)

	'''
	training_array = np.reshape(training_array, (-1, 64, 64, 3))
	print(training_array.shape)
	cv2.imshow('resized image', training_array[0])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	print(type(training_array))
	print(type(labels_to_pickle))


	pickle_file_name = 'McGill.pickle'
	file = open(pickle_file_name, 'wb')
	save = {'data': training_array,
			'labels': labels_to_pickle}
	pickle.dump(save, file)
	file.close()
	
	
	
	






		








			




if __name__ == "__main__":
    main()