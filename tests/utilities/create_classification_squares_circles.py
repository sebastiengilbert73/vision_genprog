import logging
import argparse
import ast
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--numberOfImages', help="The number of generated images. Default: 100", type=int, default=100)
parser.add_argument('--imageSizeHW', help="The image size, (H, W). Default: '(256, 256)'", default='(256, 256)')
parser.add_argument('--outputDirectory', help="The output directory. Default: '../data/squares_vs_circles/'", default='../data/squares_vs_circles/')
parser.add_argument('--objectsSize', help="The size of the objects. Default: 61", type=int, default=61)
parser.add_argument('--noiseSigma', help="The standard deviation of the white gaussian noise. Default: 5.0", type=float, default=5.0)
parser.add_argument('--backgroundGrayLevel', help="The background gray level. Default: 64", type=int, default=64)
parser.add_argument('--objectsGrayLevel', help="The objects gray level. Default: 192", type=int, default=192)
args = parser.parse_args()

image_sizeHW = ast.literal_eval(args.imageSizeHW)
if type(image_sizeHW) is not tuple:
    raise ValueError("The argument --imageSizeHW ({}) can't be converted to a tuple".format(args.imageSizeHW))
if len(image_sizeHW) != 2:
    raise ValueError("len(image_sizeHW) ({}) != 2".format(len(image_sizeHW)))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("create_classification_squares_circles.py main()")
    with open( os.path.join(args.outputDirectory, "class.csv"), 'w+' ) as class_file:
        class_file.write("image,class\n")
        for imageNdx in range(args.numberOfImages):
            noise = np.random.normal(loc=0, scale=args.noiseSigma, size=image_sizeHW)
            image = np.ones(image_sizeHW, dtype=np.uint8) * args.backgroundGrayLevel
            center = (np.random.randint(0, image_sizeHW[1]), np.random.randint(0, image_sizeHW[0]))
            classNdx = None
            if imageNdx%2 == 0:  # circle
                cv2.circle(image, center, args.objectsSize//2, args.objectsGrayLevel, thickness=cv2.FILLED)
                classNdx = 0
            else:
                cv2.rectangle(image, (center[0] - args.objectsSize//2, center[1] - args.objectsSize//2),
                              (center[0] + args.objectsSize // 2, center[1] + args.objectsSize // 2),
                              args.objectsGrayLevel, thickness=cv2.FILLED)
                classNdx =1
            image = (image.astype(float) + noise).astype(np.uint8)
            image_filename = "image_" + str(imageNdx) + '.png'
            image_filepath = os.path.join(args.outputDirectory, image_filename)
            cv2.imwrite(image_filepath, image)
            class_file.write("{},{}\n".format(image_filename, classNdx))


if __name__ == '__main__':
    main()
