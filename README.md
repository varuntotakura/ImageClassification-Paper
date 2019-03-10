# ImageClassification

Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 4800)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               2458112   
_________________________________________________________________
dense_1 (Dense)              (None, 128)               65664     
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_3 (Dense)              (None, 15)                1935      
=================================================================
Total params: 2,542,223
Trainable params: 2,542,223
Non-trainable params: 0
_________________________________________________________________

10/10 [==============================] - 0s 7ms/step
C:\Users\VARUN\Anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
  return f(*args, **kwds)
# Test accuracy: 0.8999999761581421
# Predictions :  download
# Actual :  download
[Finished in 12.3s]
