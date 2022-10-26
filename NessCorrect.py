

import cv2
import numpy as np
import mediapipe
import time



def getColor(x, w, y, h, frame):
    blueLower = (85, 100, 100)
    blueUpper = (135, 255, 255)
    kare = frame[y:y+h, x:x+w]
    #cv2.imwrite("box.jpg", kare)
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

    maske = cv2.inRange(hsv, blueLower, blueUpper)
    cv2.imwrite("maske1.jpg", maske)
    maske = cv2.dilate(maske, None, iterations=2)
    cv2.imwrite("maske_dilated.jpg", maske)
    maske = cv2.erode(maske, None, iterations=2)
    cv2.imwrite("maske2.jpg", maske)

    color_ratio = (maske.sum() // 255) / (kare.shape[0] * kare.shape[1])
    #print(maske.sum() // 255, kare.shape[0] * kare.shape[1])


    if color_ratio > 0.35:
        return True
    else:
        return False


def eltespit(x, width, y, height, img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hlms = hands.process(imgRGB)  # görüntüyü rgb olarak işler
    # print(hlms.multi_hand_landmarks) #çıktı olarak listenin içinde 21 adet dict görüyoruz ve bunlarda eklem yerleri oluyor.
    # bu sayılar 0-1 arasındadır ve x,y yi kameranın weight,height ile çarparsak koordinat yerini bulmuş oluruz

    h, w, channel = img.shape
    # print(height,width)
    if hlms.multi_hand_landmarks:  # none ifadesi dönmezse yani elimizi algılarsa buraya gir
        for handlandmarks in hlms.multi_hand_landmarks:  # 21 elemanı for ile tek tek gönderiyoruz
            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS) #landmarklar arası bağlantıları çiziyor
            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                if (fingerNum == 8): #işaret parmağının ucu
                    positionX, positionY = int(landmark.x * w),\
                                           int(landmark.y * h)  # X ve Y koordinat yerlerini bulunuyor
                    #print(positionX, positionY)
                    # if fingerNum==8:
                    cv2.circle(img, (positionX, positionY), 10, (0, 0, 255), thickness=cv2.FILLED)  # işaret parmağına daire cizer
        if (x < positionX < x + width) and (y < positionY < y + height):
            cv2.putText(img, "TEBRİKLER!", (x, y), font, 2, (255, 255, 255), 2)
            return True
        else:
            return False


net = cv2.dnn.readNet('custom-yolov4-tiny-detector_best.weights',
                      'custom-yolov4-tiny-detector.cfg')
classes = []
classes = []
with open("coco (1).names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

#time_function_done = 0
sure = True

mpHands = mediapipe.solutions.hands #elimizde 21 tane noktalar arasındaki bağlantıları çizdirmemizi sağlayacak

hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

mpDraw = mediapipe.solutions.drawing_utils #elde ettiğimiz noktaları kamera üzerinde çizer
while True:
    _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = [k if k > 0 else 0 for k in boxes[i]]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            #print("koordinat")
            #print((x,y), (x+w, y+h))

            if getColor(x, w, y, h, img):

                # ilk sinyal gelince süreyi alıyor.
                if sure:
                    print("calistim")
                    time_function_done = time.time()
                    sure = False

                # Sürenin üstüne 5 saniye geçene kadar elimi kontrol ediyor.
                # Eğer elim telefona değiyorsa süreyi o ana eşitler.
                if time.time() < (time_function_done + 5):
                    if eltespit(x, w, y, h, img):
                        time_function_done = time.time()

                # Eğer sinyalden sonra 5 saniye boyunca elim
                # telefona değmediyse ekranda elini vermedin yazıyor.
                # sure değişkeni True olarak değiştiriliyor.
                # Sonraki 5 saniyeyi kontrol edecek.
                else:
                    print(time_function_done, time.time())
                    cv2.putText(img, "elini vermedin", (x, y), font, 2, (255, 255, 255), 2)
                    if eltespit(x, w, y, h, img):
                        sure = True
            else:
                #print("maviler patladi")
                sure = True
    else:
        sure = True

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()




