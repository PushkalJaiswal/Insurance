import pickle
import os
import sklearn
from sklearn import linear_model

def checkDir():
    if 'models' in os.listdir('../'):
        return True
    return False

def makeDir():
    if checkDir():
        pass
    else:
        os.mkdir('../models')

def predict(model, input_df):
    # input=np.array(input_df)
    prediction = model.predict(input_df)
    #pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return prediction

# will save a model at ../models and will return the location+name of saved model
def saveModel(modelClass, name=None):
    fileName = name
    if name is None:
        fileName = 'model' + str(len(os.listdir('../models')))
    fileName += '.sav'
    pickle.dump(modelClass, open('../models/' + fileName, 'wb'))
    return '../models/' + fileName


# model will be loaded through the location of model that is returned from the
def loadModel(fileName):
    model = pickle.load(open(fileName, 'rb'))
    return model

'''def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips'''