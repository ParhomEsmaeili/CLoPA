from clopa.adaptation.training_utils.prompters.point import (
    uniform_random as point_uniform_random,
    # center as point_center
)
#TODO:
# from src.prompt_generators.heuristics.heuristic_prompt_utils.scribble import (
#     #Add scribble functions here later
    
# )

#TODO: 
# from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox import (
#     #Add bbox functions here later
# )

# from src.prompt_generators.heuristics.heuristic_prompt_utils.lasso import (
#     #Add lasso functions here later
# )

'''
This file contains a registry of functions which can be used for heuristics based prompt generation.

'''

base_prompter_registry = {
    'points':{
    'uniform_random': point_uniform_random,
    # 'center': point_center,
    } 
}