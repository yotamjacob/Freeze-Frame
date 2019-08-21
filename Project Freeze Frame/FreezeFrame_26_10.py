# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

################################################################################################
class myVideo():
    """Capture video and create indices"""

    def __init__(self,video):
        self.videoCaptured = cv2.VideoCapture(video)
        
        videoLength = int(self.videoCaptured.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.freezeIndices=[0,0,0,0]
        
        for i in range(0,4):
            self.freezeIndices[i] = int((videoLength/5)*(i+1))

################################################################################################
def auto_canny(image, sigma=0.95):
    """ User handled Canny function """   
	# compute the median of the single channel pixel intensities
    v = np.median(image)

	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
	# return the edged image
    return edged
################################################################################################
""" Video creating section """
# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes-
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current capture device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']
################################################################################################
def createMasks(myVideo):
    
    frameCount = 0
    iterations = 0
    freezedFrames=[0,0,0,0]
    masks=[0,0,0,0]
    
    while(myVideo.videoCaptured.isOpened()):
        ret, frame = myVideo.videoCaptured.read()
        if ret==True:
            frameCount = frameCount+1
            if iterations < 4:
                if frameCount == myVideo.freezeIndices[iterations]:

                    iterations = iterations+1
                    """PREPROCCESSING"""
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #create grayscale image
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)      #Use Gaussian filter for blurring the image
                    
                    edged=auto_canny(blurred)           #auto canny edge detector
    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
                    
                    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

                    th, im_th = cv2.threshold(closed, 220, 255, cv2.THRESH_BINARY);    

                    im_floodfill = im_th.copy() # Copy the thresholded image.       
                    
                    h, w = im_th.shape[:2]
                    
                    mask = np.zeros((h+2, w+2), np.uint8) # Mask used to flood filling.
    
                    cv2.floodFill(im_floodfill, mask, (0,0), 255); # Floodfill from point (0, 0)
                    
                    im_floodfill_inv = cv2.bitwise_not(im_floodfill) # Invert floodfilled image
                    im_out = im_th | im_floodfill_inv # Combine the two images to get the foreground.
                    """END PREPROCCESSING"""
                    if iterations==1:
                        freezedFrames[0]=frame  
                        masks[0]=im_out
                    elif iterations==2:
                        freezedFrames[1]=frame
                        masks[1]=im_out
                    elif iterations==3:
                        freezedFrames[2]=frame
                        masks[2]=im_out
                    elif iterations==4:
                        freezedFrames[3]=frame
                        masks[3]=im_out
            else:
                break
        else:
            break
    return freezedFrames, masks
################################################################################################
def pasteMasks(video,freezedFrames,masks):
    """after generating masks, paste them on the relevant frames and generate output video"""
    freeze1 = cv2.bitwise_not(freezedFrames[0])
    freeze2 = cv2.bitwise_not(freezedFrames[1])
    freeze3 = cv2.bitwise_not(freezedFrames[2])
    freeze4 = cv2.bitwise_not(freezedFrames[3])
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.avi', get_video_type('output.avi'), 25, get_dims(video.videoCaptured, '720p'))
    
    # paste masks on video
    frameCount = 0
    while(video.videoCaptured.isOpened()):
        ret, frame = video.videoCaptured.read()

        if ret==True:
            
            frameCount = frameCount+1
            """In this section we invert the chosen frames with the relevant masks"""
            if frameCount < video.freezeIndices[0]:
                cv2.bitwise_not(freeze1,frame, mask=masks[0])
            if frameCount < video.freezeIndices[1]:
                cv2.bitwise_not(freeze2,frame, mask=masks[1])
            if frameCount < video.freezeIndices[2]:
               cv2.bitwise_not(freeze3,frame, mask=masks[2])
            if frameCount < video.freezeIndices[3]:
               cv2.bitwise_not(freeze4,frame, mask=masks[3])

            # Write output video
            out.write(frame)
            cv2.imshow('output',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    out.release()
################################################################################################
def main():
    
    video1=myVideo('gymnastics.mp4')
    freezedFrames, masks = createMasks(video1)
    video2=myVideo('gymnastics.mp4')
    pasteMasks(video2,freezedFrames,masks) 
    
    # Release everything if job is finished
    video1.videoCaptured.release()
    video2.videoCaptured.release()
    cv2.destroyAllWindows()
################################################################################################
main()