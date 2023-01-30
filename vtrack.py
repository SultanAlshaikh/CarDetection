import sys
import cv2 as cv
import numpy as np

def main():
    area_cam1 = np.array([[287,133], [229,130],[210,624], [476,366]], np.int32)
    kf = cv.KalmanFilter()
    car_cascade = cv.CascadeClassifier('cars.xml')
    cap = cv.VideoCapture('city_traffic_01.mp4')
    if not cap.isOpened():
        print('Failed to read video file/ camera', file=sys.stderr)
    fps = cap.get(cv.CAP_PROP_FPS)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    recorder = cv.VideoWriter('result.avi', cv.VideoWriter_fourcc(*'MJPG'),10, (width,height))
    i = 3260
    while cap.isOpened():
        frame = 0
        i += 1
        #kf.correct(frame)
        ret, frame = cap.read()
        frame = cv.resize(frame, (640, 480))
        if not ret:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cv.polylines(frame, [area_cam1], True, (255,0,0))
        cars = car_cascade.detectMultiScale(frame_gray, 1.1, 5)
        for (x, y, w, h) in cars:
            if len(frame[y:y+h, x:x+w]) != 0:
                pass
                #cv.imwrite(f'data/car_dataset3/{i}.png', frame[y:y+h, x:x+w])
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),1)
        #kf.predict(frame)
        recorder.write(frame)
        cv.imshow("Test1", frame)
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    exit(main())