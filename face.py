import numpy as np
import cv2
import pickle 
import os 
#Սրանք հաարյան դասակարգիչներն են որոնք օգտագործվում են դեմքի բացահայտման համար
face_cascade = cv2.CascadeClassifier('scr/cascade/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('scr/cascade/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('scr/cascade/data/haarcascade_smile.xml')
eye_glasses = cv2.CascadeClassifier('scr/cascade/data/haarcascade_eye_tree_eyeglasses.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()#լոկալ բինար հիստոգրամը որպես ճանաչող ֆունկցիա
recognizer.read("scr/trainner.yml")#ներբեռնում ենք ուսուցանված տվյալները
labels = {"person_name":1}#յուրաքանչյուր լեյբլի տալիս ենք id
with open("scr/labels.pickle", 'rb') as f:#կարդում ենք լեյբլները
 og_labels = pickle.load(f)
 labels = {v:k for k,v in og_labels.items()}# տեղերով փոխում ենք, որ առաջինը մարդու անունը լինի
cap = cv2.VideoCapture(0)#համակարգչի տեսախցիկից վիդեոն է վերադարցնում

while(True):
    ret, frame = cap.read()#ֆրեյմ առ ֆրեյմ կարդացվել է
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#պատկերը վերածվել մոխրագույնի
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)#ներբեռնված նկարից օբյեկտների բացահայտման  համար
    for(x,y,w,h) in faces:#իտերացվում է դեմքերով
        #print(x,y,w,h)
        cv2.equalizeHist(gray)#ներբեռնված նկարի համար նորմալիզացնում է կոնտրաստը փիքսելների
        roi_gray = gray[y:y+h, x:x+w]#մոխրագույն պատկերի համար հետաքրքրության միջակայքը,որտեղ դեմքի կոորդինատներն են
        roi_color = frame[y:y+h, x:x+w]#նույնը գունավորի համար
        id_, conf = recognizer.predict(roi_gray)#կատարվում են կանխատեսումներ
        if conf>=50 and conf <=70:   #  եթե հետևյալ միջակայքում է դեմքը ճանաչել է     
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1,color,stroke,cv2.LINE_AA)#
        else : 
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "unknown"
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1,color,stroke,cv2.LINE_AA)
        print(conf)
        img_item = "12.jpg"
        cv2.imwrite(img_item, roi_gray)

        color = (0,255,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)#ուղղանկյուն ենք գծում դեմքի հատվածում
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        #smile =smile_cascade.detectMultiScale(roi_gray)
        #for(sx,sy,sw,sh) in smile:
            #cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,255,0), 2)
        glasses = eye_glasses.detectMultiScale(roi_gray)
        for(gx,gy,gw,gh) in glasses:
            cv2.rectangle(roi_color, (gx,gy), (gx+gw, gy+gh), (0,255,0), 2)


       
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


   