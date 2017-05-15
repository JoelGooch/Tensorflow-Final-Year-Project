import numpy as np
import scipy
import scipy.io as sio
import cv2
import sys
import mask_analysis

np.set_printoptions(threshold=np.inf)

def main():

	mask_analyser = mask_analysis.BinaryMaskAnalyser()

	target_image_size = 64
	num_channels = 3

	root_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/McGill/McGillFaces_public_v2/'


	for person in range (1, 61):

		if person < 10:
			person_string = '0' + str(person)
		else: person_string = str(person)

		if person == 18 or person == 21 or person == 23 or person == 24 or person ==26 or person == 27 or person == 35 or person == 40 or person == 45 or person == 49 or person == 52 or person ==57 or person == 60:
			additional_num = '0'
		else: additional_num ='1'

		for image in range (1, 300):

			if image < 100:
				image_string = '0' + str(image)
				if image < 10:
					image_string = '0' + image_string
			else: image_string = str(image)

			#print(image_string)

			
			image_path = root_directory + person_string + '/Image/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
			#print(image_path)
			image = cv2.imread(image_path)

			if image is not None:

				mask_path = root_directory + person_string + '/Mask/' + person_string + '_' + additional_num + '_' + image_string + '_mask.pgm'
				#print(mask_path)
				mask = cv2.imread(mask_path)

				co_ords = mask_analyser.returnMaxAreaRectangle(mask)
				x = co_ords[0]
				y = co_ords[1]
				w = co_ords[2]
				h = co_ords[3]


				if (w == h):
					longest_side = w
				elif (w > h):
					longest_side = w
				elif (h > w):
					longest_side = h


				cropped_img = image[y:y+longest_side, x:x+longest_side]
				resized_img = cv2.resize(cropped_img, (64, 64), interpolation = cv2.INTER_AREA)


				'''
				cv2.imshow('cropped image', cropped_img)
				cv2.imshow('mask', mask)
				cv2.imshow('image', image)
				print(cropped_img.shape)

				
				cv2.imshow('resized image', resized_img)
				print(resized_img.shape)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				'''

				img_save_path = root_directory + '/Final/' + person_string + '_' + additional_num + '_' + image_string + '.jpg'
				
				#cv2.imwrite(mask_save_path, mask)
				cv2.imwrite(img_save_path, resized_img)



				'''
				co_ords = mask_analyser.returnMaxAreaCircle(mask)
				x = co_ords[0]
				y = co_ords[1]
				rad = co_ords[2]

				print(x)
				print(y)
				print(rad)

				cropped_img = image[y-rad:y+rad, x-rad:x+rad]
				cv2.imshow('croppedximage', cropped_img)
				print(cropped_img.shape)
				cv2.waitKey(0)
				cv2.destroyAllWindows()	
				'''			



			




if __name__ == "__main__":
    main()