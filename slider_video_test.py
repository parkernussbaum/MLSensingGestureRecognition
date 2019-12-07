import numpy as np
import cv2
import ipdb
import pickle



# pickle open slider regression  and predict values
clf = pickle.load (open("slider.p", "rb" ))
regression_model = pickle.load (open("sliderreg.p", "rb" ))
############


cap = cv2.VideoCapture(0)
#ipdb.set_trace()
i=0
reading=[]
data = []
while(True):
    
    # Capture frame-by-frame - slider 16:9
    ret, frame = cap.read()
    frame= cv2.rectangle(frame,(400,200),(880,470),(0,255,0),2)
    #roi = im[y1:y2, x1:x2]
    roi = frame[200:470, 400:880]


    toDisplay = frame
    data.append(toDisplay)
    
    
    ##predict seciton ##  
    
    def Harshedge(k):                               # first part of pipeline
        gray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY) # convert to grayscale
        blur = cv2.blur(gray, (1, 1)) # blur the image
        ret, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY)
        return thresh
    
    out=[]
    y=0
    fv=[]
    k=[]
    i=0
    total_feature=[]

    img=roi
    y = cv2.resize(img,(40,22)) # resizing images #40 x 40

    #y = Master_List[62]
    edges = cv2.Canny(y,30,200)
    fv = np.reshape(edges,(880,1))
    fv = fv[:,0]
    #print(len(fv))
    #out.append(fv) 
    k = Harshedge(y)
    fv1 = np.reshape(k,(880,1))
    fv1 = fv1[:,0]
    for v in range(0,len(fv)):
        total_feature.append(fv[v])
    for v in range(0,len(fv1)):
        total_feature.append(fv1[v])
    out.append(total_feature) 
    total_feature=[]
    
    clf.predict(out)

    result = str(clf.predict(out))
   # result1 = str(regression_model.predict(out))

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,result,(100,100), font, 4, (0,255,0), 3, cv2.LINE_AA)
    #cv2.putText(frame,result1,(200,200), font, 4, (255,0,0), 3, cv2.LINE_AA)

    ## End of ML predict secion ##
    
    
    ##camera showing section##
    toDisplay = frame
    data.append(toDisplay)
    cv2.imshow('frame',toDisplay)
    ## camera done section ##

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
 
#edge detection
#blob detection