import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Model Loader
def load_s3fd(frozen_graph_filename,  name="", graph=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    # Then, we import the graph_def into a new Graph and returns it 
    if graph is None:
        graph = tf.Graph() 
    with graph.as_default():
        
        tf.graph_util.import_graph_def(graph_def, name=name)
    return graph

def s3fd(frozen_model = 's3fd.pb', sess=None):
    tf.compat.v1.reset_default_graph()
    
    if sess is None:
        sess = tf.compat.v1.Session()

    graph = load_s3fd(frozen_model, graph=tf.compat.v1.get_default_graph())
                                                                      
    def det_func(img): 
        outputs={}
        output_tensors = ('S3FD_slim/outputs/scores:0', 
                          'S3FD_slim/outputs/bboxes:0')
        if len(img.shape)==3:
            img = np.expand_dims(img, axis=0)
        outputs['scores'], outputs['bboxes'] = sess.run(output_tensors, {'S3FD/inputs:0': img})
        return outputs
    return det_func


# Pre-processing
def resize_with_pad(img, shape=[640,640]):
    w,h = shape

    im_h,im_w,_ = img.shape
    factor = float(w)/im_w

    new_w, new_h = (int(im_w*factor), int(im_h*factor))

    out = np.zeros(shape=[h,w,3], dtype=np.uint8)
    if new_h<h: #pad vertical
        pad = int((h-new_h)/2)

        resized = cv2.resize(img, (new_w, new_h))
        out[pad:pad+new_h,:,:] = resized
    else: #pad_horizontal
        factor = float(h)/im_h
        new_w, new_h = (int(im_w*factor), int(im_h*factor))   
        pad = int((w-new_w)/2)   

        resized = cv2.resize(img, (new_w, new_h))
        out[:,pad:pad+new_w,:] = resized

    return cv2.resize(out, (w,h))

img = plt.imread('/content/drive/My Drive/sample1.jpg')
img = resize_with_pad(img)

plt.imshow(img) #menampilkan gambar

detector = s3fd('s3fd.pb')
outputs = detector(img)

from collections import namedtuple
Prediction = namedtuple('Prediction', ['score','bbox'])
def prediction_formatting(scores, bboxes):
    predictions = []
    
    for j in range(bboxes.shape[0]):
        y1,x1,y2,x2 = bboxes[j]
        if (x2-x1>.0) and(y2-y1>0.) and scores[j]>0:
            predictions.append(Prediction(scores[j], [x1,y1,x2,y2]))
            
    return predictions

def visualize(img, objects, min_score=0.1):
    out = img.copy()
    im_h, im_w, _ = img.shape
    
    for obj in objects:
        if obj.score<min_score:
            continue
        x1,y1,x2,y2 = obj.bbox
        
        #convert to absolute
        x1 = int(x1*im_w)
        y1 = int(y1*im_h)
        x2 = int(x2*im_w)
        y2 = int(y2*im_h)
            
        w = x2-x1
        h = y2-y1
        
        c = (0,255,0)
        out = cv2.rectangle(out,(x1,y1),(x2,y2),c,2)
        
            
    return out

faces = prediction_formatting(outputs['scores'], outputs['bboxes'])              
vis = visualize( img, faces, min_score=0.8)

plt.figure(figsize=(15,15))
plt.imshow(vis)
