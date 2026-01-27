import os 
import sys 
from math import ceil 

#NOTE: With static criteria, it is probably inevitable that there is some inefficiency in terms of using samples. For example, as we reach the end of a 
#data stream, we might have a situation where we have 3 samples left but our criterion is to adapt every 5 samples. In this case, we will never be able
#to trigger adaptation again. This is a trade-off we have to accept for the sake of simplicity. More complex criteria could be designed to mitigate this, but they would also
#introduce more complexity and possibly hyperparameters to tune. 

#TODO: This is left to future versions!



class AdaptationTriggerCriterionRegistry:
    '''
    This is a registry class which holds different adjustment criteria functions.
    '''
    def __init__(
        self,
        criterion_config: dict,
        data_splitting_config: dict
    ):
        criteria = {
            'static_fraction_interval': static_fraction_interval_criterion,
            'static_raw_interval': static_raw_interval_criterion
        }

        data_splitting_strategies = {
            'static_hold_out': static_holdout_data_split,
        }
       
        if criterion_config.get('name') not in criteria:
            raise ValueError(f'Invalid trigger criterion: {criterion_config.get("name")}. Supported criteria are: {criteria}')
        elif data_splitting_config.get('name') not in data_splitting_strategies:
            raise ValueError(f'Invalid data splitting strategy: {data_splitting_config.get("name")}. Supported strategies are: {data_splitting_strategies}')
        else:
            self.criteria = criteria
            self.criterion_config = criterion_config
            self.data_splitting_strategies = data_splitting_strategies
            self.data_splitting_config = data_splitting_config

    def batch_size_pass_check(
        self,
        num_available_samples: int,
        ) -> bool:
        '''
        This is a function which does a preliminary check (before the actual criterion check) to see whether
        we have enough samples to satisfy minimum batch size requirements after data splitting.
        '''

        min_batch_size = 2
        if num_available_samples >= min_batch_size:
            return True
        elif num_available_samples == 0:
            return False 
        else:
            return False
        
    def data_split_pass_check(
        self,
        unassigned_samples: list,
        all_samples: list 
        ):
        '''
        This is a function which does a preliminary check (before the actual criterion check) to see whether
        the data splitting requirements are satisfied to even consider triggering an adaptation. 
        
        For example, in a holdout data splitting strategy we might want to ensure that we have enough samples 
        to be able to split into train and val sets according to a data splitting configuration.'''

        strategy = self.data_splitting_config.get('name', None)
        #NOTE: Fill in with other possibly required pre-checks in future versions.
        if strategy in self.data_splitting_strategies:
            return self.data_splitting_strategies[strategy](
                len(unassigned_samples),
                len(all_samples),
                self.data_splitting_config
            )
    def adaptation_criterion( 
        self,
        meta_algorithm_state: dict,
        ) -> bool:
        '''
        This is a function which determines whether the adaptation process should be triggered or not.
        
        Looks through a registry of trigger criteria and calls the appropriate function and does
        this depending on the criterion name provided. 
        '''
        
        if self.criterion_config.get('name') == 'static_fraction_interval':
            return static_fraction_interval_criterion(
                meta_algorithm_state['dataset_info']['num_samples'], 
                meta_algorithm_state['unassigned_samples'],
                fraction_interval=self.criterion_config.get('fraction')
                )
        elif self.criterion_config.get('name') == 'static_raw_interval':
            return static_raw_interval_criterion(
                meta_algorithm_state['unassigned_samples'],
                raw_samples=self.criterion_config.get('adaptation_raw_interval')
            ) 
        else:
            raise NotImplementedError(f'Adjustment criterion {self.criterion_config.get("name")} is not implemented yet.')

    def __call__(
        self, 
        meta_algorithm_state: dict
        ) -> bool:
        '''
        This is a callback which checks whether the adaptation process should be triggered or not.
        '''
    
        #First, we check whether the data splitting requirements are satisfied.
        data_split_passed, data_split_config, num_available_train_samples = self.data_split_pass_check(
            meta_algorithm_state['unassigned_samples'],
            meta_algorithm_state['memory_buffer_disk'])
        if data_split_passed:  
            #Now lets check whether we even have enough samples for a minimum batch size. 
            batch_size_passed = self.batch_size_pass_check(
                num_available_train_samples,
            )
        else:
            batch_size_passed = False #Just set to false if data split check failed.
            
        if data_split_passed and batch_size_passed:
            #If it passes this, then we can go ahead and check the actual adaptation criterion.
            return self.adaptation_criterion(meta_algorithm_state), data_split_config
        else:
            return False, data_split_config
      
    
def static_fraction_interval_criterion(
    total_samples: int, 
    unassigned_samples: list,
    fraction_interval: float) -> bool:
    '''
    This is a function which determines whether the adaptation process should be triggered or not.
    
    This just triggers adaptation every X fraction of total samples. 
    '''
    if total_samples <= 0 or total_samples == None:
        raise ValueError('total_samples must be positive integer.')
    if unassigned_samples == None:
        raise ValueError('unassigned_samples cannot be None.')
    if fraction_interval <= 0 or fraction_interval > 1:
        raise ValueError('fraction_interval must be between 0 and 1.')
    raw_interval = int(total_samples * fraction_interval)
    return len(unassigned_samples) >= raw_interval 

def static_raw_interval_criterion(
    unassigned_samples: list,
    raw_samples: int) -> bool:
    '''
    This is a function which determines whether the adaptation process should be triggered or not.
    This just triggers adaptation every X raw samples. 
    '''
    return len(unassigned_samples) >= raw_samples

def static_holdout_data_split(
    num_unassigned_samples: int,
    num_all_samples: int,
    config: dict
    ) -> bool:
    '''
    This is a function which checks whether we have enough new samples to split into train and val sets before
    even considering triggering adaptation.
    '''
    fraction_train = config.get('fraction_train')
    if fraction_train is None or not (0 < fraction_train < 1):
        raise ValueError('fraction_train must be specified in data_splitting_config and be between 0 and 1 for hold_out strategy.')
    #To be able to split into train and val sets, we need at least 1/(1-fraction_train) samples.
    min_samples_needed = int(1 / (1 - fraction_train))
    
    num_assigned_samples = num_all_samples - num_unassigned_samples 


    if num_unassigned_samples < min_samples_needed:
        
        return False, config, 0 #Not enough samples to split into train and val sets, so lets put a 0 here to definitively prevent adaptation.
    else:
        total_available_train_samples = num_assigned_samples + ceil(num_unassigned_samples * fraction_train) 
        #We will always give preference to the train split, so this is reflected in our calculation of total available train samples. Should probably
        #be an int in almost every scenario regardless since min samples needed is satisfied!  
        return True, config, total_available_train_samples