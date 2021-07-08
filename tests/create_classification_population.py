import logging
import argparse
import ast
import os
import numpy as np
import cv2
import pandas as pd
import random
import vision_genprog.classifiersPop as classifiersPop
from genprog import core as gp, evolution as gpevo
import xml.etree.ElementTree as ET
import vision_genprog.tasks.image_processing as image_processing

parser = argparse.ArgumentParser()
parser.add_argument('--imagesDirectory', help="The images directory. Default: './data/squares_vs_circles/'", default='./data/squares_vs_circles/')
parser.add_argument('--classFilename', help="The filename of the classification file. Default: 'class.csv'", default='class.csv')
parser.add_argument('--numberOfIndividuals', help="The number of individuals. Default: 50", type=int, default=50)
parser.add_argument('--levelToFunctionProbabilityDict', help="The probability to generate a function, at each level. Default: '{0: 1, 1: 1, 2: 1, 3: 0.5, 4: 0.5, 5: 0.5}'", default='{0: 1, 1: 1, 2: 1, 3: 0.5, 4: 0.5, 5: 0.5}')
parser.add_argument('--proportionOfConstants', help='The probability to generate a constant, when a variable could be used. Default: 0', type=float, default=0)
parser.add_argument('--constantCreationParametersList', help="The parameters to use when creating constants: [minFloat, maxFloat, minInt, maxInt, width, height]. Default: '[-1, 1, 0, 255, 256, 256]'", default='[-1, 1, 0, 255, 256, 256]')
parser.add_argument('--primitivesFilepath', help="The filepath to the XML file for the primitive functions. Default: './vision_genprog/tasks/image_processing.xml'", default='./vision_genprog/tasks/image_processing.xml')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)
constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)

def main():
    logging.info("create_classification_population.py main()")

    image_filepaths = ImageFilepaths(args.imagesDirectory)
    class_df = pd.read_csv(os.path.join(args.imagesDirectory, args.classFilename))
    filepathClass_list = FilepathClassList(args.imagesDirectory, class_df)

    # Split in train - validation - test
    # Shuffle the list
    random.shuffle(filepathClass_list)
    validation_start_ndx = round(0.6 * len(filepathClass_list))
    test_start_ndx = round(0.8 * len(filepathClass_list))
    train_filepathClass_list = filepathClass_list[0: validation_start_ndx]
    validation_filepathClass_list = filepathClass_list[validation_start_ndx: test_start_ndx]
    test_filepathClass_list = filepathClass_list[test_start_ndx:]

    # Create the interpreter
    primitive_functions_tree = ET.parse(args.primitivesFilepath)
    interpreter = image_processing.Interpreter(primitive_functions_tree)


    # Create a population
    classifiers_pop = classifiersPop.ClassifiersPopulation()
    classifiers_pop.Generate(
        numberOfIndividuals=args.numberOfIndividuals,
        interpreter=interpreter,
        returnType='int',
        levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
        proportionOfConstants=args.proportionOfConstants,
        constantCreationParametersList=constantCreationParametersList,
        variableNameToTypeDict={'image': 'grayscale_image'},
        functionNameToWeightDict= None
    )

    classifiers_pop.


def ImageFilepaths(images_directory):
    image_filepaths_in_directory = [os.path.join(images_directory, filename) for filename in os.listdir(images_directory)
                              if os.path.isfile(os.path.join(images_directory, filename))
                              and filename.upper().endswith('.PNG')]
    return image_filepaths_in_directory

def FilepathClassList(images_directory, class_df):
    filepathClass_list = []
    for index, row in class_df.iterrows():
        filename = row['image']
        classNdx = row['class']
        #print ("filename = {}; classNdx = {}".format(filename, classNdx))
        filepathClass_list.append((os.path.join(images_directory, filename), classNdx) )
    return filepathClass_list

if __name__ == '__main__':
    main()