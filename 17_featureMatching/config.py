import os
import shutil
import h5py

class Config:
    def __init__(self):
        
        self.device = 'cpu'
        self.device = 'cuda'
        self.device = 'tpu'
        
        if(self.device=='tpu'):
            self.storage_local_or_bucket = 'local'
            
            self.TPU_DEBUG = 0
            # self.TPU_DEBUG = 1          
        else:
            self.storage_local_or_bucket = 'local'     
        
        if( self.device == 'tpu' ):
            os.chdir( os.path.join('/', 'home', 'mfatih', 'FeatureMatchingDebug', '17_featureMatching') )
        
        self.first_experiment = 100
        
        self.use_hdf5_or_picle = 'hdf5'
        self.use_hdf5_or_picle = 'pickle'
        
        self.file_name_train = 'yfcc-sift-2000-train.hdf5'
        self.file_name_val = 'yfcc-sift-2000-val.hdf5'
        self.file_name_test = 'yfcc-sift-2000-test.hdf5'        
        
        self.input_path_bucket = '01_featureMatchingDatasets' 
        self.input_path_local = os.path.join('..', self.input_path_bucket)
        
        self.pickle_set_no = 0
        self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
        self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
        
        self.output_path_bucket = '08_featureMatchingOutputs'
        self.output_path_local = os.path.join('..', self.output_path_bucket)        
        
        self.input_channel_count = 2
        
        # self.model_type = 'CNN_Plain'
        self.model_type = 'CNN_Residual'
        # self.model_type = 'MobileNetV1'
        # self.model_type = 'MobileNetV2'
        # self.model_type = 'MobileNetV3'
        
        self.use_ratio = 0  # 0-> don't use, 1-> mask xs and ys, 2-> use as side
        self.use_mutual = 0  # 0-> don't use, 1-> mask xs and ys, 2-> use as side
        if(self.use_ratio==0 and self.use_mutual == 0):
            self.model_width = 4
        else:
            self.model_width = 6
        
        self.ess_loss = 'geo_loss'
        # self.ess_loss = 'ess_loss'
        
        self.geo_loss_ratio = 0.5
        self.ess_loss_ratio = 0.1

        # self.learning_rate = 0.001
        self.learning_rate = 0.01
        # self.learning_rate = 0.1
        
        self.n_epochs = [2, 2, 1] # always cls loss
        # self.n_epochs = [2, 0, 1] # only first chunk cls loss
        
        # training_params ->   [model_type(n_to_n, 1_to_1), N_images_in_batch, N, batch_size]
        
        # self.training_params = [ [ '1_to_1', 1, 512, 64, ],  ]        
        self.training_params = [ [ '1_to_1', 1, 512, 512, ],  ]
        # self.training_params = [ [ '1_to_1', 1, 1024, 1024, ],  ]
        # self.training_params = [ [ '1_to_1', 1, 2048, 2048, ],  ]
        
        if(self.use_hdf5_or_picle == 'hdf5'):
            
            self.num_workers = 0
            
            self.n_image_pairs_train = 541172
            self.n_image_pairs_val = 6694
            self.n_image_pairs_test = 4000
            
            # system_ram_mb = 40_000
            system_ram_mb = 10
            # system_ram_mb = 50
            # system_ram_mb = 100
            s_each_image_pair_apprx_mb = 0.05  
            s_t_image_pair_apprx_mb = self.n_image_pairs_train * s_each_image_pair_apprx_mb * ( self.num_workers + 1 )
            self.n_chunks = int( s_t_image_pair_apprx_mb / system_ram_mb ) + 1
            
        elif(self.use_hdf5_or_picle == 'pickle'):
            
            # self.num_workers = 0
            self.num_workers = 3
            
            self.n_chunks = self.get_n_chunks_from_files() 
        
        print( 'Number of chunks ' + str( self.n_chunks ) )

        self.ratio_test_th = 0.8
        
        self.obj_geod_th = 1e-4
        
        self.geo_loss_margin = 0.1

        self.bn_or_gn = 'bn'
        # self.bn_or_gn = 'gn'
        
        self.ReLu_Leaky_ReLu = 'ReLu'
        # self.ReLu_Leaky_ReLu = 'Leaky_ReLu'
        
        # self.optimizer_types = ['SGD', 'ADAM']
        self.optimizer_types = ['SGD']
        
        self.validation_chunk_or_all = 'chunk'
        self.validation_chunk_or_all = 'all'
        
        self.save_checkpoint_last_or_all = 'last'
        self.save_checkpoint_last_or_all = 'all'
        
        self.en_grad_checkpointing = False
        # self.en_grad_checkpointing = True
    
    def get_n_chunks_from_files(self):
        
        chunk = 0
        if(self.storage_local_or_bucket == 'local'):
            while True:            
                file_name_with_path = os.path.join(self.input_path_pickle_local, 'train' + f'_{chunk:04d}' + '.pkl')            
                if os.path.isfile(file_name_with_path):
                    chunk += 1
                else:
                    return chunk

        elif(self.storage_local_or_bucket == 'bucket'):                
            from google.cloud import storage
            
            storage_client = storage.Client()    
            bucket = storage_client.get_bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=self.input_path_pickle_bucket)

            blob_names = []
            for blob in blobs:
                blob_names.append(blob.name)

            while True:
                file_name = os.path.join(self.input_path_pickle_bucket, 'train' + f'_{chunk:04d}' + '.pkl')
                if(file_name in blob_names):
                    chunk += 1
                else:
                    return chunk
                
    def copy_config_file_to_output_folder(self, experiment):    
        
        folder_name = os.path.join(self.output_path_local, f'{experiment:04d}')        
        if(not os.path.isdir(folder_name)):
            os.makedirs(folder_name)        
        destination = os.path.join(folder_name, 'config.py')
        shutil.copyfile('config.py', destination)
        
        self.update_output_folder(experiment)
    
    def update_output_folder(self, experiment):
    
        self.output_path_local = os.path.join(self.output_path_local, f'{experiment:04d}')
    
    def update_training_params_for_test(self):
        
        for training_params in self.training_params:
            training_params[3] = training_params[2]

def get_config():
    return Config()
        
if __name__ == '__main__':
    
    config = Config()
