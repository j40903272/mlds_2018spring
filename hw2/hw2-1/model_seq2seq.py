
# coding: utf-8

# In[ ]:


from utils import *
from config import *


# In[ ]:


def test(data_path, output_path):
    try:
        model.load_weights('model_weights.hdf5')
    except Exception as e:
        print ('load model weight failed !')
        print (str(e))
    
    # load data
    test_id_list = open(os.path.join(data_path, 'id.txt')).read().split()
    test_data = {i:np.load(os.path.join(data_path, 'feat/')+ i + '.npy') for i in test_id_list}
    
    gen_output(test_data, test_id_list, decode_sequence_reduce, output_path)


# In[ ]:


def train():
    load_data()
    
    # preprocess includes dictionary construction & misspelling correction
    X, Y, X_test, Y_test = preprocess(test_label, train_label)
    
    #checkpoint = ModelCheckpoint("model.hdf5", monitor='val_acc', save_best_only=True)
    #earlystop = EarlyStopping(monitor='val_acc', patience=5)
    mycallback = MyCallback()
    callbacks_list = [mycallback]
    
    for epoch in range(epochs):
        
        # output will be generate on every epoch end
        print ('epoch', epoch)
        try:
            if np.random.random() < teacher_force_ratio:
                print ('teacher forcing')
                history = model.fit_generator(data_generator(X, Y), steps_per_epoch=int((len(Y)+batch_size-1)/batch_size), 
                                              validation_data=validation_generator(X_test, Y_test), 
                                              validation_steps=int((len(Y_test)+batch_size-1)/batch_size),
                                              epochs=1, callbacks=callbacks_list)
            else:
                print ('inference mode')
                history = model_inf.fit_generator(inf_data_generator(X, Y), 
                                                  steps_per_epoch=int((len(Y)+batch_size-1)/batch_size), 
                                                  validation_data=inf_validation_generator(X_test, Y_test), 
                                                  validation_steps=int((len(Y_test)+batch_size-1)/batch_size),
                                                  epochs=1, 
                                                  callbacks=callbacks_list)
        except KeyboardInterrupt:
            print ('KeyboardInterrupt')
            break


# In[ ]:


mode = sys.argv[1]
if mode == 'test':
    test(sys.argv[2], sys.argv[3])
elif mode == 'train':
    print ('training mode still got problems. See training details in ipynb files')
    #train()
else:
    print ('argv[3] should be "test" or "train"')
    exit()

