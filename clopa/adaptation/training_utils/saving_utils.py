import torch 
import os 
import shutil 

def save_checkpoint(temp_dir, filename, desired_dir, configs):
    #We will assume that temp_dir is a subdir for desired dir. 
    
    # if not os.path.exists(os.path.join(desired_dir, filename)): 
    
    #Commenting out because otherwise if we want to overwrite checkpoints this function won't work! 

    #If our temp storage dir exists from a prior interrupted run, we will clear it as we didn't finish!
    if os.path.exists(os.path.join(desired_dir, temp_dir)):
        shutil.rmtree(os.path.join(desired_dir, temp_dir))
    os.makedirs(os.path.join(desired_dir, temp_dir))
    #Storing a checkpoint in a temp dir in case of interruptions. We will only delete this dir once successfully moved
    torch.save(
        configs, os.path.join(desired_dir, temp_dir, filename))
    #We will assume that the desired dir exists.
    shutil.move(
        os.path.join(desired_dir, temp_dir, filename),
        os.path.join(desired_dir, filename)
    )

    shutil.rmtree(os.path.join(desired_dir, temp_dir))
                    