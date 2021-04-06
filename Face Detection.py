import cv2

FaceCasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
click = cv2.VideoCapture(0)

while True:
    
    _, pic = click.read()

    GrayScale = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    Face = FaceCasc.detectMultiScale(GrayScale, 1.1, 4)

    for (x, y, width, height) in Face:
        cv2.rectangle(pic, (x, y), (x+width, y+height), (0, 0, 255), 2)

    cv2.imshow('img', pic)

    key = cv2.waitKey(30) & 0xff
    if key==27:
        break
        
click.release()