import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time

from datasets.datasets import get_dataset
from datasets.datasets import collate_fn2

from models.models import get_model
from models.models import get_model_structure

from samplers.CustomBatchSampler import get_sampler

from loss_module import loss_functions
import checkpoint
import plots

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

def train_and_val(index, FLAGS):
    
    config                  = FLAGS['config']
    learning_rate           = FLAGS['learning_rate']
    n_epochs                = FLAGS['n_epochs']
    num_workers             = FLAGS['num_workers']
    model_type              = FLAGS['model_type']
    bn_or_gn                = FLAGS['bn_or_gn']
    optimizer_type          = FLAGS['optimizer_type']
    en_grad_checkpointing   = FLAGS['en_grad_checkpointing']
    input_type              = FLAGS['input_type']
    N_images_in_batch       = FLAGS['N_images_in_batch']
    N                       = FLAGS['N']
    batch_size              = FLAGS['batch_size']
       
    
    torch.manual_seed(1234)
    
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()  
        
    xm.master_print(f"Master Print by Process {index} using {xm.xla_real_devices([str(device)])[0]}")
    print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 0" )
    
    # Barrier to prevent master from exiting before workers connect.
    xm.rendezvous('init')

    model_width = config.model_width
    model = get_model( model_type, N, model_width, bn_or_gn, en_grad_checkpointing )

    print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 1" )
    
    # if xm.is_master_ordinal():
    #     device_for_model_structure = 'cpu'
    #     get_model_structure( config, device_for_model_structure, model, N, model_width, en_grad_checkpointing)        
    #     model = model.to(device)

    print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 2" )
    xm.rendezvous('get_model_structure')

    if(optimizer_type == 'ADAM'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 3" )

    checkpoint.save_initial_checkpoint( config, model, optimizer )

    xm.rendezvous('save_initial_checkpoint')

    print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4" )

    epoch, chunk, model, optimizer, success_checkpoint, \
    loss_checkpoint, proc_time_checkpoint = checkpoint.load_checkpoint( config, device, model, optimizer)

    print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 5" )

    xm.rendezvous('load_checkpoint')
      
    while epoch < n_epochs[0]:
        while chunk < config.n_chunks:
            
            loss_cls_train = 0
            loss_geo_train = 0
            loss_ess_train = 0
            loss_count_train = 0       
        
            loss_cls_val = 0
            loss_geo_val = 0
            loss_ess_val = 0
            loss_count_val = 0
                    
            confusion_matrix_at_epoch_train_device  = torch.zeros( (2,2), device = device, requires_grad = False )
            confusion_matrix_at_epoch_val_device    = torch.zeros( (2,2), device = device, requires_grad = False )
            
### Generating dataset, sampler and dataloader for the current train chunk
            
            dataset_train = get_dataset( config, input_type, N_images_in_batch, N, batch_size, train_val_test = 'train', chunk=chunk )
            
            sampler_train = get_sampler( dataset_train, input_type, N_images_in_batch, N, batch_size )
            
            dataloader_train = DataLoader(  dataset = dataset_train,
                                            sampler = sampler_train,
                                            pin_memory = True,
                                            num_workers = num_workers,
                                            collate_fn = collate_fn2, )
            
            mp_dataloader_train = pl.MpDeviceLoader( dataloader_train, device, )
            
            start_time_train = time.perf_counter()
        
            model.train()
            
            # print( 'Device is ' + xm.xla_real_devices( [  str(device)])[0] )
            # print( 'Device is ' + xm.xla_real_devices( [  str(device)])[0] + ' with ' + 
            #                                               str(xm.get_memory_info(device)['kb_free']) + ' KB free memory ' + 
            #                                               str(xm.get_memory_info(device)['kb_total']) + ' KB total memory ' )
            
            for i, data in enumerate(mp_dataloader_train):
                
                optimizer.zero_grad()
                
                xs_device = data['xs'].to(device)
                labels_device = data['ys'].to(device)
                
                xs_ess =  data['xs_ess'].to(device)
                R_device =  data['R'].to(device)
                t_device =  data['t'].to(device)
                virtPt_device =  data['virtPt'].to(device)
                
                logits = model(xs_device)
                
                classif_loss = loss_functions.get_losses( config, device, labels_device, logits)
                
                geo_loss, ess_loss, _ = loss_functions.calculate_ess_loss_and_L2loss( config, logits, xs_ess, R_device, t_device, virtPt_device )
                                
                if( epoch < config.n_epochs[1] or ( epoch == config.n_epochs[1] and chunk < config.n_epochs[2] ) ):
                    loss = classif_loss                
                else:
                    if(config.ess_loss == 'geo_loss'):
                        loss = 1 * classif_loss + config.geo_loss_ratio * geo_loss
                    elif(config.ess_loss == 'ess_loss'):
                        loss = 1 * classif_loss + config.ess_loss_ratio * ess_loss
                
                loss.backward()

                xm.optimizer_step(optimizer)
                    
                confusion_matrix_at_epoch_train_device[0,0] += torch.sum( torch.logical_and( logits<0, labels_device>config.obj_geod_th ) )
                confusion_matrix_at_epoch_train_device[0,1] += torch.sum( torch.logical_and( logits>0, labels_device>config.obj_geod_th ) )
                confusion_matrix_at_epoch_train_device[1,0] += torch.sum( torch.logical_and( logits<0, labels_device<config.obj_geod_th ) )
                confusion_matrix_at_epoch_train_device[1,1] += torch.sum( torch.logical_and( logits>0, labels_device<config.obj_geod_th ) )
                        
                loss_cls_train = loss_cls_train * loss_count_train + classif_loss.detach().cpu().numpy() * batch_size
                loss_ess_train = loss_ess_train * loss_count_train + ess_loss.detach().cpu().numpy() * batch_size
                loss_geo_train = loss_geo_train * loss_count_train + geo_loss.detach().cpu().numpy() * batch_size
                loss_count_train = loss_count_train + batch_size
                loss_cls_train = loss_cls_train / loss_count_train
                loss_ess_train = loss_ess_train / loss_count_train  
                loss_geo_train = loss_geo_train / loss_count_train
                    
                if( ( (i*batch_size) % 100000 ) > ( ((i+1)*batch_size) % 100000 ) or (i+1) == len(dataloader_train) ):
                    
                    tot_it_train = torch.sum(confusion_matrix_at_epoch_train_device)
                    acc_train = torch.sum(confusion_matrix_at_epoch_train_device[0,0]+confusion_matrix_at_epoch_train_device[1,1]) / tot_it_train
                    pre_train = confusion_matrix_at_epoch_train_device[1,1] / torch.sum(confusion_matrix_at_epoch_train_device[:,1])
                    rec_train = confusion_matrix_at_epoch_train_device[1,1] / torch.sum(confusion_matrix_at_epoch_train_device[1,:])
                    f1_train = 2 * pre_train * rec_train / ( pre_train + rec_train )
                            
                    xm.master_print("Train Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} lCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                    # xm.master_print("Train Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} lCls {:.6f}  CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                            .format(    epoch,
                                        n_epochs[0]-1,
                                        chunk,
                                        config.n_chunks-1,
                                        i,
                                        len(dataloader_train)-1,
                                        learning_rate,
                                        loss_cls_train,
                                        loss_geo_train,
                                        loss_ess_train,
                                        int(torch.sum(confusion_matrix_at_epoch_train_device[0,0]+confusion_matrix_at_epoch_train_device[1,1])),
                                        int(tot_it_train),
                                        acc_train,
                                        pre_train,
                                        rec_train,
                                        f1_train,
                                        ) )
            

            success_checkpoint[0, epoch, chunk, :] = np.array([acc_train.detach().cpu().numpy(), pre_train.detach().cpu().numpy(), rec_train.detach().cpu().numpy(), f1_train.detach().cpu().numpy()])
            loss_checkpoint[0, epoch, chunk, :] = np.array([loss_cls_train, loss_geo_train, loss_ess_train])                
            proc_time_checkpoint[0,epoch, chunk] = time.perf_counter() - start_time_train

            xm.rendezvous('shift_train_results')

### Generating dataset, sampler and dataloader for the current validation chunk

            dataset_val = get_dataset( config, input_type, N_images_in_batch, N, batch_size, train_val_test = 'val', chunk=chunk )
            
            sampler_val = get_sampler( dataset_val, input_type, N_images_in_batch, N, batch_size )
                        
            dataloader_val = DataLoader(    dataset = dataset_val,
                                            sampler = sampler_val,
                                            pin_memory = True,
                                            num_workers = num_workers,
                                            collate_fn = collate_fn2, )
            
            mp_dataloader_val = pl.MpDeviceLoader( dataloader_val, device, )                                   
            
            start_time_val = time.perf_counter()
            
            model.eval()  # Sets the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(mp_dataloader_val):
                    
                    xs_device = data['xs'].to(device)
                    labels_device = data['ys'].to(device)
                    
                    xs_ess =  data['xs_ess'].to(device)
                    R_device =  data['R'].to(device)
                    t_device =  data['t'].to(device)
                    virtPt_device =  data['virtPt'].to(device) 
                        
                    logits = model(xs_device)
                    
                    classif_loss = loss_functions.get_losses( config, device, labels_device, logits)
                    
                    geo_loss, ess_loss, _ = loss_functions.calculate_ess_loss_and_L2loss( config, logits, xs_ess, R_device, t_device, virtPt_device )

                    confusion_matrix_at_epoch_val_device[0,0] += torch.sum( torch.logical_and( logits<0, labels_device>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_val_device[0,1] += torch.sum( torch.logical_and( logits>0, labels_device>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_val_device[1,0] += torch.sum( torch.logical_and( logits<0, labels_device<config.obj_geod_th ) )
                    confusion_matrix_at_epoch_val_device[1,1] += torch.sum( torch.logical_and( logits>0, labels_device<config.obj_geod_th ) )
                                         
                    loss_cls_val = loss_cls_val * loss_count_val + classif_loss.detach().cpu().numpy() * batch_size
                    loss_ess_val = loss_ess_val * loss_count_val + ess_loss.detach().cpu().numpy() * N
                    loss_geo_val = loss_geo_val * loss_count_val + geo_loss.detach().cpu().numpy() * N
                    loss_count_val = loss_count_val + batch_size
                    loss_cls_val = loss_cls_val / loss_count_val
                    loss_ess_val = loss_ess_val / loss_count_val  
                    loss_geo_val = loss_geo_val / loss_count_val
                            
                    if( ( (i*batch_size) % 100000 ) > ( ((i+1)*batch_size) % 100000 ) or (i+1) == len(dataloader_val) ):
                        
                        tot_it_val = torch.sum(confusion_matrix_at_epoch_val_device)
                        acc_val = torch.sum(confusion_matrix_at_epoch_val_device[0,0]+confusion_matrix_at_epoch_val_device[1,1]) / tot_it_val
                        pre_val = confusion_matrix_at_epoch_val_device[1,1] / torch.sum(confusion_matrix_at_epoch_val_device[:,1])
                        rec_val = confusion_matrix_at_epoch_val_device[1,1] / torch.sum(confusion_matrix_at_epoch_val_device[1,:])
                        f1_val = 2 * pre_val * rec_val / ( pre_val + rec_val )
                        
                        xm.master_print("Val Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} LossCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                        # xm.master_print("Test Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} LossCls {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                                .format(    epoch,
                                            n_epochs[0]-1,
                                            chunk,
                                            config.n_chunks-1,
                                            i,
                                            len(dataloader_val)-1,
                                            learning_rate,
                                            loss_cls_val,
                                            loss_geo_val,
                                            loss_ess_val,
                                            int(torch.sum(confusion_matrix_at_epoch_val_device[0,0]+confusion_matrix_at_epoch_val_device[1,1])),
                                            int(tot_it_val),
                                            acc_val,
                                            pre_val,
                                            rec_val,
                                            f1_val,
                                            ) )                        
            
            success_checkpoint[1, epoch, chunk, :] = np.array([acc_val.detach().cpu().numpy(), pre_val.detach().cpu().numpy(), rec_val.detach().cpu().numpy(), f1_val.detach().cpu().numpy()])
            loss_checkpoint[1, epoch, chunk, :] = np.array([loss_cls_val, loss_geo_val, loss_ess_val])
            proc_time_checkpoint[1,epoch, chunk] = time.perf_counter() - start_time_val

            xm.rendezvous('shift_val_results')

            if xm.is_master_ordinal():
                plots.plot_success_and_loss( config, epoch, chunk, success_checkpoint, loss_checkpoint)  
                
                plots.plot_proc_time( config, epoch, chunk, proc_time_checkpoint)
                
                checkpoint.save_checkpoint( config, epoch, chunk, model, optimizer, success_checkpoint, loss_checkpoint, proc_time_checkpoint)
            
            xm.rendezvous('update_plots_and_checkpoints')

            if(chunk==config.n_chunks-1):
                chunk = 0
                epoch = epoch + 1
                break
            else:
                chunk = chunk + 1
            
            xm.master_print("-" * 40)

            xm.rendezvous('end_of_chunk')
            
    return 0
