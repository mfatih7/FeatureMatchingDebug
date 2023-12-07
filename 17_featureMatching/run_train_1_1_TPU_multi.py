from config import get_config

from  train_1_1_each_sample_in_single_batch_TPU_multi import train_and_val

import torch_xla.distributed.xla_multiprocessing as xmp

import tpu_related.set_env_variables_for_TPU as set_env_variables_for_TPU

# Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again
# (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader
# instance creation logic here, as it doesn’t need to be re-executed in workers.

if __name__ == '__main__':

    set_env_variables_for_TPU.set_env_variables_for_TPU_PJRT( )

    config = get_config()
    
    set_env_variables_for_TPU.set_env_debug_variables_for_TPU_PJRT( config )
    
    experiment = config.first_experiment
    
    config.copy_config_file_to_output_folder( experiment )
    
    # learning_rate = config.learning_rate
    # n_epochs = config.n_epochs
    # num_workers = config.num_workers
    # model_type = config.model_type
    # bn_or_gn = config.bn_or_gn
    # optimizer_type = config.optimizer_types[0]
    # en_grad_checkpointing = config.en_grad_checkpointing
    input_type = config.training_params[0][0]
    N_images_in_batch = config.training_params[0][1]
    N = config.training_params[0][2]
    batch_size = config.training_params[0][3]
    
    if(input_type=='1_to_1'):
        
        if( N_images_in_batch >= 1 and N == batch_size ):
            
            print('Training starts for ' + 'train_1_1_each_sample_in_single_batch')
            
            FLAGS = {}
            FLAGS['config']                     = config
            FLAGS['learning_rate']              = config.learning_rate
            FLAGS['n_epochs']                   = config.n_epochs
            FLAGS['num_workers']                = config.num_workers
            FLAGS['model_type']                 = config.model_type
            FLAGS['bn_or_gn']                   = config.bn_or_gn
            FLAGS['optimizer_type']             = config.optimizer_types[0]
            FLAGS['en_grad_checkpointing']      = config.en_grad_checkpointing
            FLAGS['input_type']                 = config.training_params[0][0]
            FLAGS['N_images_in_batch']          = config.training_params[0][1]
            FLAGS['N']                          = config.training_params[0][2]
            FLAGS['batch_size']                 = config.training_params[0][3]
            
            xmp.spawn(train_and_val, args=(FLAGS,) )
        else:
            raise ValueError(f"The provided arguments are not valid: {input_type} {N_images_in_batch} {N} {batch_size}")
    else:
        raise ValueError(f"The provided argument is not valid: {input_type}")
    