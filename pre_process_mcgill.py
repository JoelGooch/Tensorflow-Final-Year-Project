import numpy as np
import scipy
import scipy.io as sio
import cv2
import sys
import mask_analysis

# this file handles the pre processing of McGill dataset, it cycles all images and gathers the co ordinates of the corresponding mask
# it uses these co-ordinates on the original image and crops it, it then resizes this to 64x64x3

def main():

	mask_analyser = mask_analysis.BinaryMaskAnalyser()

	target_image_size = 64
	num_channels = 3

	root_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/McGill/McGillFaces_public_v2/'


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

		for image in range (1, 301):

			# image files have image number included in format 001, 002.. 021, 022... 101, 102. this adds two 0s to single digits, and 1 to double
			if image < 100:
				image_string = '0' + str(image)
				if image < 10:
					image_string = '0' + image_string
			else: image_string = str(image)

			# read the current image
			image_path = root_directory + person_string + '/Image/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
			image = cv2.imread(image_path)

			# not all images are present, so if this file read returned nothing, skip step
			if image is not None:

				# read in corresponding mask
				mask_path = root_directory + person_string + '/Mask/' + person_string + '_' + additional_num + '_' + image_string + '_mask.pgm'
				mask = cv2.imread(mask_path)

				# return co ordinates using mask analysis from Patacchiola DeepGaze module
				co_ords = mask_analyser.returnMaxAreaRectangle(mask)
				x = co_ords[0]
				y = co_ords[1]
				w = co_ords[2]
				h = co_ords[3]

				# need a squared subframe so take longest dimension
				if (w == h):
					longest_side = w
				elif (w > h):
					longest_side = w
				elif (h > w):
					longest_side = h

				# crop and resize image to suit
				cropped_img = image[y:y+longest_side, x:x+longest_side]
				resized_img = cv2.resize(cropped_img, (64, 64), interpolation = cv2.INTER_AREA)

				# save new cropped image in new location
				img_save_path = root_directory + '/Final/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
				cv2.imwrite(img_save_path, resized_img)


if __name__ == "__main__":
    main()