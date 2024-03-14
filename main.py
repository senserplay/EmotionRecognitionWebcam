import cv2
from deepface import DeepFace

def highlightFace(net, frame, conf_threshold=0.7):
    global emotion
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            face = frame[y1-25:y2+25, x1-25:x2+25]
            is_written = cv2.imwrite('face.png', face)
            try:
                analysis = DeepFace.analyze('face.png',actions = ['emotion'])
                emotion=analysis[0]['dominant_emotion']
            except:
                pass
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            cv2.putText(frameOpencvDnn, emotion, (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),thickness=5)
    return  frameOpencvDnn, faceBoxes

if __name__ == "__main__":
    #Загружаем веса для распознавания лиц
    faceProto = "opencv_face_detector.pbtxt"
    #Конфигурация самой нейросети
    faceModel = "opencv_face_detector_uint8.pb"
    #Запускаем нейросеть
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    #Получаем видео с вебкамеры
    video = cv2.VideoCapture(0)
    emotion="neutral"
    #Остановка работы при нажатии любой кнопки
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        cv2.imshow("Face detection", resultImg)
