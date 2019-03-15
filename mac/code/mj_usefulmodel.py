
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, concatenate, Activation, ReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam
from keras import regularizers
from keras import initializers
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
import keras.backend as K
import numpy as np

#m5 = MyKeras(deepth=4, dp_rate=0.5, size=256, bn=True, activation='prelu',
#  lr=0.002, l2=0,epochs=5,batch_size=16,standard=True)

#一个keras常用包装器
class MyKeras(BaseEstimator):
    classes_=np.array([0,1])
    def __init__(self,deepth= 4,dp_rate=0.5,size=256,bn=True,activation='relu',lr=0.002,l2=0.001,
                 epochs=5,batch_size=16,standard=True):
        self.deepth=deepth
        self.dp_rate=dp_rate
        self.size=size
        self.bn=bn
        self.activation=activation
        self.lr=lr
        self.l2=l2
        self.epochs=epochs
        self.batch_size=batch_size
        def make_model1():

            initializer = initializers.glorot_uniform()
            regularizer = regularizers.l2(self.l2)
            model = Sequential()
            model.add(BatchNormalization())
            for i in range(self.deepth):
                model.add(Dense(self.size, kernel_initializer=initializer,
                                kernel_regularizer=regularizer,
                                activity_regularizer=regularizer))
                if self.bn:
                    model.add(BatchNormalization(), )
                if self.activation=='relu':
                    model.add(ReLU())
                elif self.activation=='prelu':
                    model.add(PReLU())
                model.add(Dropout(self.dp_rate), )

            model.add(Dense(1, activation='sigmoid'))
            op = Nadam(lr=self.lr)
            model.compile(optimizer=op,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            return model
        if standard:
            self.m = make_pipeline(StandardScaler(), KerasClassifier(make_model1))
        else:
            self.m = make_pipeline(MinMaxScaler(), KerasClassifier(make_model1))

    def fit(self,x,y):
        from imblearn.over_sampling import RandomOverSampler,SMOTE
        x3, y3 = RandomOverSampler().fit_sample(x, y)

        self.m.fit(x3, y3, kerasclassifier__class_weight='auto', kerasclassifier__batch_size=self.batch_size
               , kerasclassifier__verbose=0,kerasclassifier__epochs=self.epochs)
        return self
    def predict_proba(self,x):
        p=self.m.predict_proba(x)
        #print('closing')

        return p



from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# x3=np.vstack([x1] + 15*[ x1[y1==1] ] )
# y3=np.hstack([y1] + 15*[ y1[y1==1] ] )
from keras.layers import Conv2D,Dense,Activation,MaxPool2D,Flatten,Dropout,Conv1D,MaxPool1D,AvgPool1D,\
    GlobalMaxPooling1D,GlobalAveragePooling1D,Input
from keras.models import Model

#自己写的CNN（没啥用
class MyCNN(BaseEstimator):
    classes_=np.array([0, 1])

    def __init__(self):
        def make_cnn():

            inputs= Input( shape=(184,1) )
            #print(inputs)
            cv1=Conv1D(32,7,strides=1,padding='same') ( inputs )
            #cv1=BatchNormalization() (cv1)
            cv1=Activation('relu') ( cv1    )
            #cv1= Dropout(0.1) (cv1  )

            cv2 = Conv1D(32, 30, strides=1,padding='same')(inputs)
            #cv2 = BatchNormalization()(cv2)
            cv2 = Activation('relu')(cv2)
            #cv2 = Dropout(0.1)(cv2)

            cv3 = Conv1D(32, 14, strides=1, padding='same')(inputs)
            #cv3 = BatchNormalization()(cv3)
            cv3 = Activation('relu')(cv3)
            #cv3 = Dropout(0.1)(cv3)

            cv=concatenate([cv1,cv2,cv3])
            cv=Dropout(0.3)(cv)
            x1=GlobalMaxPooling1D()(cv)
            x2=MaxPool1D(4)(cv)
            x2=Flatten()(x2)
            #m.add()
            #m.add(Flatten())
            x=concatenate([x1,x2])
            x=Dropout(0.3)(x)
            #x = BatchNormalization()(x)
            x=Dense(256, activation='relu' ,)(x)
            x=BatchNormalization()(x)
            x=Dropout(0.2)(x)
            x=Dense(1)(x)
            x=BatchNormalization()(x)
            x=Activation('sigmoid')(x)
            #注意：后面不能乱加逗号，会被当成是tuple
            m = Model(inputs=inputs,outputs=x)
            m.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            #m.compile(loss='hinge',optimizer='adam')
            return m
        self.m = KerasClassifier(make_cnn)
    def fit(self,x,y):
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        x3, y3 = RandomOverSampler().fit_sample(x, y)
        x3=x3.reshape(-1,184,1)
        self.m.fit(x3, y3, class_weight='auto', batch_size=16)
                   #, kerasclassifier__verbose=0)
        return self
    def predict_proba(self,x):
        p=self.m.predict_proba(x.reshape(-1,184,1))
        return p
