import numpy as np
from sklearn import svm

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


import cv2
from PIL import Image
from skimage.feature import hog
import os
import dill

car_dataset_dir_large = 'data/cars_dataset/'
car_dataset_dir = 'data/car_dataset2/'
car_dataset_filenames = list()
car_hog_features = list()

neg_dataset_dir = 'data/natural_images/'
neg_subdirs = ['airplane/', 'cat/', 'fruit/', 'person/', 'dog/', 'flower/']
neg_dataset_filenames = list()
neg_hog_features = list()

def calc_hog_cars():
    # TODO: Cars are arranged by make. So we have a bias to those at top.
    # Correct by reading files names, and shuffle randomly to reduce bias.
    
    i = 0
    for f in os.listdir(car_dataset_dir):
        i += 1
        img = np.asarray(Image.open(car_dataset_dir+f))
        img = cv2.cvtColor(cv2.resize(img,(96,64)),
                        cv2.COLOR_RGB2GRAY)
        hog_feat = hog(img,
                    pixels_per_cell=(16,16),
                    cells_per_block=(2,2))                        
        car_hog_features.append(hog_feat)
    
    fi = open('car_hog2.pickle', 'wb')
    dill.dump(car_hog_features, fi)
    fi.close()
    print(f'Done calculating Hog for Pos for {i} samples')

def calc_hog_neg():
    k = 81 # 81 for each class in neutral
    i = 0
    j = 0
    for sub_neg in neg_subdirs:
        i = 0
        for f in os.listdir(neg_dataset_dir+sub_neg):
            if i == k: continue
            i += 1
            j += 1
            img = np.asarray(Image.open(neg_dataset_dir+sub_neg+f))
            img = cv2.cvtColor(cv2.resize(img,(96,64)),cv2.COLOR_RGB2GRAY)
            neg_hog_feat = hog(img,
                    pixels_per_cell=(16,16),
                    cells_per_block=(2,2))
            neg_hog_features.append(neg_hog_feat)

    fi = open('neg_hog2.pickle', 'wb')
    dill.dump(neg_hog_features, fi)
    fi.close()
    print(f'Done calculating Hog for Neg for {j} samples')

def svm_train():
    pos_labels = list(np.ones(len(car_hog_features)))
    neg_labels = list(np.zeros(len(neg_hog_features)))

    # Aggregate cars and others into x
    x = np.asarray(car_hog_features + neg_hog_features)
    y = np.asarray(pos_labels + neg_labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                        test_size=0.3,
                                        random_state=1337)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    grid_search_params = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['rbf']}
    ]
    svm_classifer = GridSearchCV(svm.SVC(), grid_search_params, refit=True).fit(x_train, y_train)
    cross_val = cross_val_score(svm_classifer, x, y, cv=5)
    print(f'RBF 5 fold cross validation scores {cross_val}')
    fi = open('svm_cars2.pickle', 'wb')
    dill.dump(svm_classifer, fi)
    fi.close()

def svm_test():
    f = open('svm_cars2.pickle', 'rb')
    svm_classifier : svm.SVC = dill.load(f)
    k = 2000
    i = 0
    cars_hog = list()
    for f in os.listdir(car_dataset_dir_large):
        i += 1
        if i < 1000: continue
        img = np.asarray(Image.open(car_dataset_dir_large+f))
        img = cv2.cvtColor(cv2.resize(img,(96,64)),
                        cv2.COLOR_RGB2GRAY)
        hog_feat = hog(img,
                    pixels_per_cell=(16,16),
                    cells_per_block=(2,2))                        
        cars_hog.append(hog_feat) 
        if i == k: break
    x_test1 = np.asarray(cars_hog)
    y_pred = svm_classifier.predict(x_test1)
    print(y_pred)
    i = 0
    for p in y_pred:
        if int(p) == 1:
            i+=1
    print(f'Got {i} correct')
    
    k = 2000
    i = 0
    neg_hog = list()
    for sub_neg in neg_subdirs:
        for f in os.listdir(neg_dataset_dir+sub_neg):
            i += 1
            if i < 1000: continue
            img = np.asarray(Image.open(neg_dataset_dir+sub_neg+f))
            img = cv2.cvtColor(cv2.resize(img,(96,64)),cv2.COLOR_RGB2GRAY)
            neg_hog_feat = hog(img,
                    pixels_per_cell=(16,16),
                    cells_per_block=(2,2))
            neg_hog.append(neg_hog_feat)
            if i == k: break
        if i == k: break
    x_test2 = np.asarray(neg_hog)
    y_pred = svm_classifier.predict(x_test2)
    print(y_pred)
    i = 0
    for p in y_pred:
        if  p < 1:
            i+=1
    print(f'Got {i} correct')
    x_test1_labels = list(np.ones(len(cars_hog[:500])))
    x_test2_labels = list(np.zeros(len(neg_hog[:500])))
    x_test = np.asarray(cars_hog[:500] + neg_hog[:500])
    y_test = np.asarray(x_test1_labels + x_test2_labels)
    print(x_test.shape)
    print(y_test.shape)

    score = svm_classifier.score(x_test, y_test)
    cross_val = cross_val_score(svm_classifier, x_test, y_test, cv=10)
    print(f'SVM score {score}')
    print(f'SVM 10-fold cross-val {cross_val}')
    

def svm_is_car(filename, svm_classifier: svm.SVC):
    img = Image.open(filename)
    img = np.asarray(img)
    img = cv2.cvtColor(cv2.resize(img, (96,64)), cv2.COLOR_RGB2GRAY)
    hog_features = hog(img,
                       pixels_per_cell=(16,16),
                       cells_per_block=(2,2))
    y_pred = svm_classifier.predict(np.asarray([hog_features]))
    return y_pred


def load_svm_classifier(classifier_name):
    f = open(classifier_name, 'rb')
    svm_classifier : svm.SVC = dill.load(f)
    return svm_classifier


def main():
    # Options to approach this is to:
    #   - Use slinding window (kernel) and apply the svm classifier, and combine
    #       close boxes.
    #   - Use movement backgroundSub and apply the svm classifer to verify cars
    #       leads to non-detection of cars not moving... maybe.
    # calc_hog_cars()
    # calc_hog_neg()
    # svm_train()
    #svm_test()
    svm_class = load_svm_classifier('svm_cars2.pickle')
    is_car = svm_is_car('car1.png', svm_class)
    print(int(is_car[0]))
    # is_car = svm_is_car('car2.png', svm_class)
    # print(is_car)
    # is_car = svm_is_car('car3.png', svm_class)
    # print(is_car)
    # is_car = svm_is_car('car4.png', svm_class)
    # print(is_car)
    # is_car = svm_is_car('nocar1.png', svm_class)
    # print(is_car)
    # is_car = svm_is_car('nocar2.png', svm_class)
    # print(is_car)
    # is_car = svm_is_car('nocar3.png', svm_class)
    # print(is_car)
    

if __name__ == '__main__':
    exit(main())