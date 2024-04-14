from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image as KivyImage
import cv2
import mediapipe as mp

p1,p2,p3,p4,p5,p6,p7,p8=0,0,0,0,0,0,0,0


import cv2
import mediapipe as mp
import math
import csv
import numpy as np
from keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score

count=0
arre=[]
# Define function to calculate angle
def calculate_angle(l,h,p,p1, p2, p3,ct):
    
    # Convert landmarks to numpy arrays
    p1 = np.array([p1.x, p1.y])
    p2 = np.array([p2.x, p2.y])
    p3 = np.array([p3.x, p3.y])
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle (in degrees) between the vectors
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    if ct[0]==1:
        if h>angle>l:
            landmark_colors[p]=(0, 255, 0)
        elif h+20>angle>l-20:
            landmark_colors[p]=(0, 165, 255)
        else:
            
            ct.append(landmark_colors_points[p])
                # print("Correct : ",landmark_colors_points[p])        
                # print("Correct : ",arre)        
            landmark_colors[p]=(0, 0, 255)


    return angle

import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()



import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

landmark_colors_points={
    11: "LEFT SHOULDER" ,             # Red (RIGHT_EYE_INNER)
    12: "RIGHT SHOULDER" ,             # Red (RIGHT_EYE_INNER)
    13: "LEFT ELBOW"      ,        # Red (RIGHT_EYE_INNER)
    14:"RIGHT ELBOW"       ,       # Red (RIGHT_EYE_INNER)
    23: "LEFT HIP"          ,    # Red (RIGHT_EYE_INNER)
    24: "RIGHT HIP"          ,    # Red (RIGHT_EYE_INNER)
    25: "LEFT KNEE"           ,   # Red (RIGHT_EYE_INNER)
    26: "RIGHT KNEE" ,

}

# Define colors for specific landmarks
landmark_colors = {
    11: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    12: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    13: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    14: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    23: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    24: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    25: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
    26: (0, 0, 255),              # Red (RIGHT_EYE_INNER)
}

distances_data=[]




import numpy as np
from joblib import load
# Step 1: Load the trained model
model_filename = "jan_model.joblib"  # Change this to the path of your saved model
rf_classifier = load(model_filename)



import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to draw landmarks on the image
def draw_landmarks(image, landmarks):
    # Draw landmarks with specified colors
    for landmark_index, landmark_color in landmark_colors.items():
        landmark_point = landmarks.landmark[landmark_index]
        if landmark_point.visibility > 0.5:
            x, y = int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])
            cv2.circle(image, (x, y), 10, landmark_color, -1)
    
    # Draw connecting lines between landmarks in white
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_index = connection[0]
        end_index = connection[1]
        if landmarks.landmark[start_index].visibility > 0.5 and landmarks.landmark[end_index].visibility > 0.5:
            start_point = (int(landmarks.landmark[start_index].x * image.shape[1]), int(landmarks.landmark[start_index].y * image.shape[0]))
            end_point = (int(landmarks.landmark[end_index].x * image.shape[1]), int(landmarks.landmark[end_index].y * image.shape[0]))
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)




class PostureDetectionApp(App):
    posenamep=''
    arre = []
    p1,p2,p3,p4,p5,p6,p7,p8=0,0,0,0,0,0,0,0
        
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        # self.label = Label(text='No posture detected')
        # self.layout.add_widget(self.label)
        self.capture = cv2.VideoCapture(0)
        self.kivy_image = KivyImage()
        self.layout.add_widget(self.kivy_image)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout


    def update(self, dt):
        posenamep=''
        # p1,p2,p3,p4,p5,p6,p7,p8
        # arre 
        count = [0]
        ret, frame = self.capture.read()
        if ret:

            frame_height, frame_width, _ = frame.shape
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect pose landmarks
            results = pose.process(frame_rgb)

            # Draw landmarks and connecting lines with specified colors
            if results.pose_landmarks :

                # if(results.pose_landmarks.landmark[23]['x'] < frame_height):
                #     print(results.pose_landmarks.landmark[23]['x'] )
                p1=calculate_angle(125,176,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                p2=calculate_angle(166,179,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                p3=calculate_angle(151,171,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                p4=calculate_angle(156,178,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                p5=calculate_angle(107,173,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                p6=calculate_angle(168,175,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                p7=calculate_angle(154,179,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                p8=calculate_angle(152,175,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                count[0]=1
                draw_landmarks(frame, results.pose_landmarks)



                x=[int(p1),int(p2),int(p3),int(p4),int(p5),int(p6),int(p7),int(p8),]
                # Step 2: Prepare the new data
                new_data = np.array([x])  # New data in the same format as the training data
                # Step 3: Make predictions on the new data
                predicted_output = rf_classifier.predict(new_data)
                posenump=predicted_output[0]+1
                print(posenump)

                if posenump==1:
                    posenamep='prayer pose'

                elif posenump==2:
                    posenamep='raised arm pose'

                elif posenump==3:
                    posenamep='hand to foot'

                elif posenump==4:
                    posenamep='equastrian pose'

                elif posenump==5:
                    posenamep='Stick Pose'

                elif posenump==6:
                    posenamep='salut with'

                elif posenump==7:
                    posenamep='Cobra Pose'

                elif posenump==8:
                    posenamep='Mountain Pose'
                else:
                    posenamep=''


                if posenump==1:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(0,176,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(3,26,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(161,179,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(137,179,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(34,61,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(0,43,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(172,179,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(159,175,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)

                    posename='prayer pose'


                elif posenump==2:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(125,176,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(166,179,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(151,171,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(156,178,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(107,173,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(168,175,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(154,179,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(152,175,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='raised arm pose'

                elif posenump==3:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(149,179,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(31,123,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(56,111,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(148,179,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(173,179,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(58,124,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(55,64,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(143,179,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='hand to foot'

                elif posenump==4:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(162,179,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(49,101,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(27,148,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(77,180,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(159,180,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(72,110,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(24,160,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(97,178,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='equastrian pose'

                elif posenump==5:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(149,179,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(24,110,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(24,177,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(121,167,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(149,176,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(62,115,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(130,178,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(144,169,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='Stick Pose'

                elif posenump==6:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(71,149,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(89,101,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(43,50,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(62,79,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(44,67,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(100,114,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(38,44,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(50,64,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='salut with'

                elif posenump==7:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(169,179,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(17,20,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(88,179,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(77,180,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(171,180,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(25,30,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(119,135,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(152,168,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='Cobra Pose'

                elif posenump==8:
                    # Calculate and print angle between landmark points 1, 2, and 3
                    p1=calculate_angle(172,179,13,results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11],count)
                    p2=calculate_angle(134,141,11,results.pose_landmarks.landmark[13], results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23],count)
                    p3=calculate_angle(48,56,23,results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25],count)
                    p4=calculate_angle(163,176,25,results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27],count)
                    p5=calculate_angle(170,177,14,results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12],count)
                    p6=calculate_angle(133,142,12,results.pose_landmarks.landmark[14], results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24],count)
                    p7=calculate_angle(51,60,24,results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26],count)
                    p8=calculate_angle(163,178,26,results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28],count)
                    posename='Mountain Pose'



                xxx=','.join(count[1:])

                cv2.putText(frame, xxx, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



            # cv2.putText(frame, 'pred :'+posenamep, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if posenamep:
                cv2.putText(frame, 'pose:'+posenamep, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


            self.display_frame(frame)

    def draw_landmarks(self, frame, landmarks):
        # Draw landmarks with specified colors
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def display_frame(self, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.kivy_image.texture = texture1

if __name__ == '__main__':
    PostureDetectionApp().run()
