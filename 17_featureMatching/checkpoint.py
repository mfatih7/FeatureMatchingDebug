import os
import shutil
import numpy as np
import torch

# Checkpoint Functions For Train-Val

def save_initial_checkpoint( config, model, optimizer ):

    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints' )
    if(os.path.exists(checkpoint_path)==0):
        os.makedirs(checkpoint_path)
        
        success_checkpoint = np.zeros( (2, config.n_epochs[0], config.n_chunks, 4) )
        loss_checkpoint = np.zeros( (2, config.n_epochs[0], config.n_chunks, 3) )
        proc_time_checkpoint = np.zeros( (2, config.n_epochs[0], config.n_chunks) )
        
        checkpoint = {
                      'epoch' : 0,
                      'chunk' : 0,
                      
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                      
                      'success_checkpoint': success_checkpoint,
                      'loss_checkpoint': loss_checkpoint,
                      'proc_time_checkpoint': proc_time_checkpoint,
                     }
        checkpoint_file_with_path = os.path.join(checkpoint_path, 'model.pth.tar')
        torch.save(checkpoint, checkpoint_file_with_path )

def load_checkpoint( config, device, model, optimizer ):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints' )
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'model.pth.tar')
    
    checkpoint = torch.load(checkpoint_file_with_path)
    
    epoch = checkpoint['epoch']
    chunk = checkpoint['chunk']
    
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.to(device)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    success_checkpoint = checkpoint['success_checkpoint']
    loss_checkpoint = checkpoint['loss_checkpoint']
    proc_time_checkpoint = checkpoint['proc_time_checkpoint']
    
    return epoch, chunk, model, optimizer, success_checkpoint, loss_checkpoint, proc_time_checkpoint

def save_checkpoint( config, epoch, chunk, model, optimizer, success_checkpoint, loss_checkpoint, proc_time_checkpoint):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints' )
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'model.pth.tar')
    if(config.save_checkpoint_last_or_all == 'all'):
        archive_checkpoint_file_with_path = os.path.join(checkpoint_path, 'model' + f'_{epoch:04d}_{chunk:04d}' + '.pth.tar')    
    elif(config.save_checkpoint_last_or_all == 'last'):
        archive_checkpoint_file_with_path = os.path.join(checkpoint_path, 'model' + '_prev' + '.pth.tar')   
    shutil.copyfile(checkpoint_file_with_path, archive_checkpoint_file_with_path) 
    
    if(chunk==config.n_chunks-1):
        chunk = 0
        epoch = epoch + 1
    else:
        chunk = chunk + 1
    
    checkpoint = {
                  'epoch' : epoch,
                  'chunk' : chunk,
                  
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                  
                  'success_checkpoint': success_checkpoint,
                  'loss_checkpoint': loss_checkpoint,
                  'proc_time_checkpoint': proc_time_checkpoint,
                 }       
    torch.save(checkpoint, checkpoint_file_with_path )

# Checkpoint Functions For Test

def load_test_checkpoint( config, device, model, checkpoint_file_with_path):
    
    # if( os.path.exists( os.path.join(config.output_path_local, 'checkpoints', 'model.pth.tar') ) ):
    #     checkpoint_file_with_path = os.path.join(config.output_path_local, 'checkpoints', 'model.pth.tar')
    # else:
    #     assert(0)
    
    checkpoint = torch.load(checkpoint_file_with_path)
    
    print( 'Loading checkpoint file ' + checkpoint_file_with_path )
    
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.to(device)
    
    return model

def update_mAP_checkpoint(mAP_checkpoint, err_q, err_t, err_qt, epoch, chunk):
    
    if(mAP_checkpoint.shape[0]==2):        
        assert(0)        
    else:        
        for err_ind, error in enumerate( [err_q, err_t, err_qt] ):            
            mAP_checkpoint[ 0, epoch, chunk, err_ind, int(np.ceil(error)-1) ] += 1    
    return mAP_checkpoint

def get_all_checkpoint_files_for_test(config):
    checkpoint_files = []
    
    for filename in os.listdir( os.path.join( config.output_path_local, 'checkpoints' ) ):
        filename_with_path = os.path.join(config.output_path_local, 'checkpoints', filename)
        if os.path.isfile(filename_with_path):
            checkpoint_files.append(filename_with_path)
    
    elements_to_move = [ os.path.join( config.output_path_local, 'checkpoints', 'model_prev.pth.tar'),\
                         os.path.join( config.output_path_local, 'checkpoints', 'model.pth.tar') ]
    for element_to_move in elements_to_move:
        if element_to_move in checkpoint_files:
            checkpoint_files.remove(element_to_move)
            checkpoint_files.append(element_to_move)
            
    return checkpoint_files
    
def save_test_checkpoint( config, success_checkpoint, loss_checkpoint, proc_time_checkpoint, mAP_checkpoint):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints_test' )
    if(os.path.exists(checkpoint_path)==0):
        os.makedirs(checkpoint_path)
    
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'checkpoint_test.pth.tar')    
    
    checkpoint = {                  
                  'success_checkpoint': success_checkpoint,
                  'loss_checkpoint': loss_checkpoint,
                  'proc_time_checkpoint': proc_time_checkpoint,
                  'mAP_checkpoint': mAP_checkpoint,
                 }       
    torch.save(checkpoint, checkpoint_file_with_path )