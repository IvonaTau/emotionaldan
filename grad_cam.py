from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

emotionDict7 = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

IMGSIZE = 224


def visualize(batch_img, batch_label, modelPath, dan, img_mask=1, batch_size=1,img_size=224):
    images = tf.placeholder(tf.float32,[None, IMGSIZE,IMGSIZE,1])
    labels = tf.placeholder(tf.float32, [None, ])
    
    # gradient for partial linearization. We only care about target visualization class. 
    y_c = tf.reduce_sum(tf.multiply(dan['S2_Emotion'], labels), axis=1)
    
    # Get last convolutional layer gradient for generating gradCAM visualization
    target_conv_layer = dan['S2_Conv4b']
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

    with tf.Session() as sess:    
        Saver = tf.train.Saver()
        Saver.restore(sess, modelPath)
    
        prob = sess.run([dan['Pred_emotion']], 
                    {dan['InputImage']:batch_img,                   
                         dan['S1_isTrain']:False,
                         dan['S2_isTrain']:False}) 
    
        target_conv_layer_value, target_conv_layer_grad_value = sess.run(
        [target_conv_layer, target_conv_layer_grad], 
        feed_dict={images: batch_img,
                    labels: batch_label,
                    dan['InputImage']:batch_img,
                    dan['S1_isTrain']:False,
                    dan['S2_isTrain']:False})
    
        print('Predicted emotion:', emotionDict7[prob[0][0]])
        print('True emotion', emotionDict7[batch_label[0]] )
    
        for i in range(batch_size):
            visualize_points(batch_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], img_mask=img_mask)
    

def visualize_points(image, conv_output, conv_grad, img_mask = 1,img_size=224):
    """
    img_mask - threshold for showing original image instead of cam activations"""
    output = conv_output           
    grads_val = conv_grad   
    # print("grads_val shape:", grads_val.shape)
    # print("output shape:", output.shape)

    weights = np.mean(grads_val, axis = (0, 1)) 
    # print('weights shape', weights.shape)
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)
    

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (img_size,img_size), preserve_range=True)
    
    print(cam.shape)

    img = image.astype(float)
    img = img - np.min(img)
    img = img / np.max(img)

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)     
    
    fig = plt.figure(figsize=(12, 16))    
    ax = fig.add_subplot(221)
    img = np.reshape(img, (img_size, img_size))
    imgplot = plt.imshow(img, cmap='gray')
    ax.set_title('Input Image')
    
    for n in range(0, img_size):
        for m in range(0,img_size):
            # when to show original image
#             if cam_heatmap[n][m][0] < red_thres or cam_heatmap[n][m][1] > yellow_thres:
            if cam[n][m] < img_mask:
                
                cam_heatmap[n][m][0] = np.uint8(img[n][m]*255)
                cam_heatmap[n][m][1] = np.uint8(img[n][m]*255)
                cam_heatmap[n][m][2] = np.uint8(img[n][m]*255)
    
    fig = plt.figure(figsize=(12, 16))    
    ax = fig.add_subplot(222)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')    
    
    plt.show()