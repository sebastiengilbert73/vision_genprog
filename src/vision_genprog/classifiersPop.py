import genprog.evolution as gpevo
import genprog
from typing import Dict, List, Any, Set, Optional, Union, Tuple


class ClassifiersPopulation(gpevo.Population):
    def __init__(self):
        pass

    def EvaluateIndividualCosts(self, inputOutputTuplesList: List[ Tuple[ Dict[str, Any], Any ] ],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: genprog.core.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float) -> Dict[genprog.core.Individual, float]:
        individual_to_cost = {}
        if len(inputOutputTuplesList) == 0:
            raise ValueError("classifiersPop.ClassifiersPopulation.EvaluateIndividualCosts(): len(inputOutputTuplesList) == 0")
        for individual in self._individualsList:
            cost_sum = 0
            for inputOutput in inputOutputTuplesList:
                variableName_to_value = inputOutput[0]
                target_class_index = inputOutput[1]
                predicted_class_index = interpreter.Evaluate(individual, variableNameToTypeDict,
                                                             variableName_to_value, 'int')
                if predicted_class_index != target_class_index:
                    cost_sum += 1
            individual_to_cost[individual] = cost_sum / len(inputOutputTuplesList)
