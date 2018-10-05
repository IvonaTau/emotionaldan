import os
import time
import datetime

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

from layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer
from utils import cyclic_learning_rate


def NormRmse(GroudTruth, Prediction, n_landmark=68):
    Gt = tf.reshape(GroudTruth, [-1, n_landmark, 2])
    Pt = tf.reshape(Prediction, [-1, n_landmark, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
    return loss/norm


def augment(images, labels, labels_em):
    brght_imgs = tf.image.random_brightness(images, max_delta=0.3)
    cntrst_imgs = tf.image.random_contrast(brght_imgs, lower=0.2 , upper=1.8)
    # hue_imgs = tf.image.random_hue(cntrst_imgs, max_delta=0.2)
    return cntrst_imgs, labels, labels_em

def emoDAN(MeanShapeNumpy, batch_size, nb_emotions=7, lr_stage1=0.001, lr_stage2=0.001, n_landmark=68, IMGSIZE=224):

    InputImage = tf.placeholder(tf.float32,[None, IMGSIZE,IMGSIZE,1])
    GroundTruth = tf.placeholder(tf.float32,[None, n_landmark * 2])
    Emotion_Labels = tf.placeholder(tf.int32, [None, ])
    
   
    # dataset = tf.data.Dataset.from_tensor_slices((x, y, z)).batch(batch_size)
    # iter_ = dataset.make_initializable_iterator()
    # InputImage, GroundTruth, Emotion_Labels = iter_.get_next()
    
    MeanShape = tf.constant(MeanShapeNumpy, dtype=tf.float32)
    S1_isTrain = tf.placeholder(tf.bool)
    S2_isTrain = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, trainable=False)
    Ret_dict = {}
    Ret_dict['InputImage'] = InputImage
    Ret_dict['GroundTruth'] = GroundTruth
    Ret_dict['Emotion_labels'] = Emotion_Labels
    # Ret_dict['x'] = x
    # Ret_dict['y'] = y
    # Ret_dict['z'] = z

    
    Ret_dict['S1_isTrain'] = S1_isTrain
    Ret_dict['S2_isTrain'] = S2_isTrain
    
    
    InputImage, GroundTruth, Emotion_Labels = augment(InputImage,GroundTruth,Emotion_Labels)

    with tf.variable_scope('Stage1'):

        S1_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(InputImage,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv1a,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool1 = tf.layers.max_pooling2d(S1_Conv1b,2,2,padding='same')

        S1_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool1,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv2a,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool2 = tf.layers.max_pooling2d(S1_Conv2b,2,2,padding='same')

        S1_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool2,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv3a,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool3 = tf.layers.max_pooling2d(S1_Conv3b,2,2,padding='same')

        S1_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool3,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv4a,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool4 = tf.layers.max_pooling2d(S1_Conv4b,2,2,padding='same')

        S1_Pool4_Flat = tf.contrib.layers.flatten(S1_Pool4)
        S1_DropOut = tf.layers.dropout(S1_Pool4_Flat,0.5,training=S1_isTrain)

        S1_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S1_DropOut,256,activation=tf.nn.relu,\
            kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain,name = 'S1_Fc1')
        S1_Fc2 = tf.layers.dense(S1_Fc1,n_landmark * 2)

        S1_Ret = S1_Fc2 + MeanShape
        S1_Cost = tf.reduce_mean(NormRmse(GroundTruth, S1_Ret))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(lr_stage1).minimize(S1_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))
        
    Ret_dict['S1_Ret'] = S1_Ret
    Ret_dict['S1_Cost'] = S1_Cost
    Ret_dict['S1_Optimizer'] = S1_Optimizer

    with tf.variable_scope('Stage2'):

        S2_AffineParam = TransformParamsLayer(S1_Ret, MeanShape)
        S2_InputImage = AffineTransformLayer(InputImage, S2_AffineParam)
        S2_InputLandmark = LandmarkTransformLayer(S1_Ret, S2_AffineParam)
        S2_InputHeatmap = LandmarkImageLayer(S2_InputLandmark)

        S2_Feature = tf.reshape(tf.layers.dense(S1_Fc1,int((IMGSIZE / 2) * (IMGSIZE / 2)),\
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),(-1,int(IMGSIZE / 2),int(IMGSIZE / 2),1))
        S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(IMGSIZE,IMGSIZE),1)

        S2_ConcatInput = tf.layers.batch_normalization(tf.concat([S2_InputImage,S2_InputHeatmap,S2_FeatureUpScale],3),\
            training=S2_isTrain)
        S2_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(S2_ConcatInput,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv1a,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Pool1 = tf.layers.max_pooling2d(S2_Conv1b,2,2,padding='same')

        S2_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool1,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv2a,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Pool2 = tf.layers.max_pooling2d(S2_Conv2b,2,2,padding='same')

        S2_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool2,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv3a,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Pool3 = tf.layers.max_pooling2d(S2_Conv3b,2,2,padding='same')

        S2_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool3,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv4a,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Pool4 = tf.layers.max_pooling2d(S2_Conv4b,2,2,padding='same')

        S2_Pool4_Flat = tf.contrib.layers.flatten(S2_Pool4)
        S2_DropOut = tf.layers.dropout(S2_Pool4_Flat,0.5,training=S2_isTrain)

        S2_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S2_DropOut,256,\
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        S2_Fc2 = tf.layers.dense(S2_Fc1,n_landmark * 2)
        
        S2_Emotion = tf.layers.dense(S2_Fc1, nb_emotions)
        Pred_Emotion = tf.nn.softmax(S2_Emotion)
        S2_Pred_Emotion = tf.argmax(input=Pred_Emotion, axis=1)
        
        correct_prediction = tf.equal(Emotion_Labels, tf.cast(S2_Pred_Emotion, tf.int32))
        emotion_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        S2_Ret = LandmarkTransformLayer(S2_Fc2 + S2_InputLandmark,S2_AffineParam, Inverse=True)
        S2_Cost_landm = tf.reduce_mean(NormRmse(GroundTruth,S2_Ret)) # cost for landmarks
        
        one_hot_labels = tf.one_hot(indices=tf.cast(Emotion_Labels, tf.int32), depth=nb_emotions)
        print_output = tf.Print(S2_Pred_Emotion, [Pred_Emotion, Emotion_Labels, S2_Pred_Emotion], summarize=100000)
        S2_Cost_emotion = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits = S2_Emotion)) # loss for emotion prediction
        Joint_Cost = 0.5*S2_Cost_landm + 0.5*S2_Cost_emotion
        
        learning_rate = cyclic_learning_rate(global_step,
                                             learning_rate = 0.0001,
                                             max_lr = 0.05,
                                             step_size = 10000)
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage2')):
            S2_Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(Joint_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage2"), global_step=global_step)

    Ret_dict['S2_Ret'] = S2_Ret
    Ret_dict['S2_Cost'] = S2_Cost_landm
    Ret_dict['S2_Optimizer'] = S2_Optimizer
    
    Ret_dict['Joint_Cost'] = Joint_Cost
    Ret_dict['Emotion_Accuracy']=emotion_accuracy
    Ret_dict['Pred_emotion']= S2_Pred_Emotion

    Ret_dict['S2_InputImage'] = S2_InputImage
    Ret_dict['S2_InputLandmark'] = S2_InputLandmark
    Ret_dict['S2_InputHeatmap'] = S2_InputHeatmap
    Ret_dict['S2_FeatureUpScale'] = S2_FeatureUpScale
    
    Ret_dict['S2_Conv4b'] = S2_Conv4b
    Ret_dict['S2_Emotion'] = S2_Emotion
    
    Ret_dict['lr'] = learning_rate
    
    return Ret_dict