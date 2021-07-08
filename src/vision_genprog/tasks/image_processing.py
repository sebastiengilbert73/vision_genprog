import genprog.core as gp
import genprog.evolution as gpevo
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import numpy as np
import vision_genprog.utilities

possible_types = ['grayscale_image', 'color_image', 'binary_image',
                  'float', 'int', 'bool']

class Interpreter(gp.Interpreter):

    def FunctionDefinition(self, functionName: str, argumentsList: List[Any]) -> Any:
        pass

    def CreateConstant(self, returnType: str, parametersList: Optional[ List[Any] ] ) -> str:
        if returnType == 'grayscale_image':
            if len(parametersList) < 6:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 6".format(
                        returnType, len(parametersList)))
            black_img = np.zeros((parametersList[5], parametersList[4]), dtype=np.uint8)
            return vision_genprog.utilities.ArrayToString(black_img)
        elif returnType == 'color_image':
            if len(parametersList) < 6:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 6".format(
                        returnType, len(parametersList)))
            black_img = np.zeros((parametersList[5], parametersList[4], 3), dtype=np.uint8)
            return vision_genprog.utilities.ArrayToString(black_img)
        elif returnType == 'binary_image':
            if len(parametersList) < 6:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 6".format(
                        returnType, len(parametersList)))
            black_img = np.zeros((parametersList[5], parametersList[4]), dtype=np.uint8)
            return vision_genprog.utilities.ArrayToString(black_img)
        elif returnType == 'float':
            if len(parametersList) < 2:
                raise ValueError("image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 2".format(returnType, len(parametersList)))
            value = np.random.uniform(parametersList[0], parametersList[1])
            return str(value)
        elif returnType == 'int':
            if len(parametersList) < 4:
                raise ValueError(
                    "image_processing.Interpreter.CreateConstant(): Creating a '{}': len(parametersList) ({}) < 4".format(
                        returnType, len(parametersList)))
            value = np.random.randint(parametersList[2], parametersList[3] + 1)
            return str(value)
        elif returnType == 'bool':
            if np.random.randint(0, 2) == 0:
                return 'true'
            else:
                return 'false'
        else:
            raise NotImplementedError("image_processing.Interpreter.CreateConstant(): Not implemented return type '{}'".format(returnType))

    def PossibleTypes(self) -> List[str]:
        return possible_types