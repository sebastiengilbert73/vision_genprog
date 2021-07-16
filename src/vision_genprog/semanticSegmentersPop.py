import genprog.evolution as gpevo
import genprog
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import numpy as np
import cv2

class SemanticSegmentersPopulation(gpevo.Population):
    def __init__(self):
        pass

    def EvaluateIndividualCosts(self, inputOutputTuplesList: List[Tuple[Dict[str, Any], Any]],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: genprog.core.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float) -> Dict[genprog.core.Individual, float]:
        individual_to_cost = {}
        if len(inputOutputTuplesList) == 0:
            raise ValueError("semanticSegmentersPop.SemanticSegmentersPopulation.EvaluateIndividualCosts(): len(inputOutputTuplesList) == 0")
        for individual in self._individualsList:
            cost_sum = 0
            for inputOutput in inputOutputTuplesList:
                intersection_over_union = self.IntersectionOverUnion(individual, inputOutput, variableNameToTypeDict,
                                                                     interpreter, returnType)
                cost_sum += 1 - intersection_over_union
            individual_to_cost[individual] = cost_sum / len(inputOutputTuplesList)

        if weightForNumberOfElements != 0:
            for individual in self._individualsList:
                individual_to_cost[individual] += weightForNumberOfElements * len(individual.Elements())

        return individual_to_cost

    def IntersectionOverUnion(self, individual, inputOutputTuple, variableNameToTypeDict, interpreter, returnType):
        variableName_to_value = inputOutputTuple[0]
        target_segmentation = inputOutputTuple[1]
        predicted_segmentation = interpreter.Evaluate(individual, variableNameToTypeDict,
                                                      variableName_to_value, returnType)
        target_non_zero_count = cv2.countNonZero(target_segmentation)
        if target_non_zero_count == 0:
            if cv2.countNonZero(predicted_segmentation) > 0:
                return 1.0
            else:
                return 0.0
        else:  # target_non_zero_count > 0
            union_mask = cv2.max(target_segmentation, predicted_segmentation)
            intersection_mask = cv2.min(target_segmentation, predicted_segmentation)
            intersection_over_union = cv2.countNonZero(intersection_mask) / cv2.countNonZero(union_mask)
            return intersection_over_union

    def BatchIntersectionOverUnion(self, individual, inputOutputTuplesList, variableNameToTypeDict, interpreter, returnType):
        intersection_over_union_list = []
        for inputOutputTuple in inputOutputTuplesList:
            intersection_over_union = self.IntersectionOverUnion(individual, inputOutputTuple, variableNameToTypeDict, interpreter, returnType)
            intersection_over_union_list.append(intersection_over_union)
        return intersection_over_union_list