import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import serial, random, time
import os, shlex, subprocess
import requests, cv2

# ESP32CAM의 IP 주소와 포트
url = 'http://192.168.60.85/'

# 얼굴 인식/분석 모델과 이미지 resize 크기 변수
loaded_model = tf.keras.models.load_model('image_classification_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128,128

# 스트리밍 시작 요청을 보내는 함수
def start_stream():
    response = requests.get(url + 'control?var=start')
    if response.status_code == 200:
        print("Streaming started successfully")
    else:
        print("Failed to start streaming")

# 현재 시간을 파일 이름으로 하는 이미지를 저장하는 함수
def save_image():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response = requests.get(url + 'capture', stream=True)
    if response.status_code == 200:
        with open(f"{timestamp}.jpg", 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {timestamp}.jpg")
    else:
        print("Failed to save image")

# 얼굴 인식, 나이 분류 함수
def detect_and_classify_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    results = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
        face = np.expand_dims(face, axis=0)
        predictions = loaded_model.predict(face)
        predicted_class = np.argmax(predictions)
        if predicted_class == 0:
            label = 'kid'
        elif predicted_class == 1:
            label = 'youth'
        elif predicted_class == 2:
            label = 'midlife'
        elif predicted_class == 3:
            label = 'old'
        results.append({'x': x, 'y': y, 'w': w, 'h': h, 'class': label})
    
    return results

# 메인 함수
def main():
    ser = serial.Serial('COM5', 9600, timeout=1)  # 포트 이름은 시스템에 따라 다를 수 있음
    time.sleep(2)  # 시리얼 포트 초기화 대기
    global flag
    flag = None
    
    try:
        while True:
            # 빨간 LED가 꺼질 예정이라는 신호를 아두이노에게 보냄
            ser.write(b"Turn off red LED\n")
            print("Sent signal to turn off red LED")
            
            # 아두이노로부터 작업 완료 신호를 기다림
            response = ser.readline().decode('utf-8').strip()
            if response == "Done":
                print("Arduino has finished its job")
            
            time.sleep(1)  # 작업 완료 신호를 받은 후 잠시 대기

            #---------------------------------사진 촬영---------------------------------------



            # 'Start Stream' 버튼 클릭
            start_stream()
            
            save_image()
            
            # 스트리밍 중지 요청
            response = requests.get(url + 'control?var=stop')
            if response.status_code == 200:
                print("Streaming stopped successfully")
            else:
                print("Failed to stop streaming")

            


           
            #---------------------------#yolov5 이미지 분석---------------------------------


            
            pic = [p for p in os.listdir('.') if '.jpg' in p]
            if pic[-1] == flag:
                pass
            else:
                command = shlex.split('python yolov5/detect.py --save-csv --weights best.pt --conf 0.5 --source '+str(pic[-1]))
                subprocess.run(command)
                flag = pic[-1]


            
            
            # ------------------- ---------아이, 노인 분석--------- -------------------- 


            # 이미지 예측 및 결과 저장
            image_path = str(pic[-1])
            results = detect_and_classify_faces(image_path)
            
            # 결과를 CSV로 저장
            results_df = pd.DataFrame(results)

            folder = [f for f in os.listdir('yolov5/runs/detect')][-1]
            chk = os.listdir('yolov5/runs/detect/'+folder)

            # 카메라에 그 어떤 사람도 찍히지 않은 경우 csv파일이 생성되지 않음. 이를 예외 처리
            if 'predictions.csv' not in chk: 
                prv_df = pd.DataFrame(data = [[str(pic[-1]), None, None]])
            else:
                prv_df = pd.read_csv('yolov5/runs/detect/'+folder+'/predictions.csv', header=None)
                
                bins = []
                for i in range(len(results_df)):
                    bins.append([str(pic[-1]), results_df['class'][i], 0])
                    
                instant_df = pd.DataFrame(bins)
                
                prv_df = pd.concat([prv_df,instant_df])

            prv_df.to_csv('yolov5/runs/detect/'+folder+'/predictions.csv', index=False)



            # ------------------- ------------------ -------------------- 
            
            idx = ['normal', 'old', 'kid', 'wheelchair', 'stroller', 'pregnant']
            cols = list(map(lambda x: '%02d~%02d' % (x,x+2),range(0,23,2)))

            # 월별,일별 데이터 취합 폴더 생성
            if 'monthly' not in os.listdir('.'):
                os.mkdir('monthly')
            if 'daily' not in os.listdir('.'):
                os.mkdir('daily')


            now = datetime.now()

            # 매일 자정 일별 데이터 정리
            if (now.strftime('%H:%M:%S')=='00:00:00') & (now.microsecond==0):
                prv_d = (now.date() - timedelta(days=1)).strftime('%Y-%m-%d')    # 어제 날짜(20??-??-??)

                 # 데이터를 취합한 자료가 이미 있다면 실행하지 않음
                if 'prediction_{}'.format(prv_d) not in os.listdir('./daily'):

                    cnt_d = pd.DataFrame(index=idx, columns=cols, data=np.zeros((6,12)))    # 데이터 취합에 사용할 객체
                    pred = [p for p in os.listdir('.') if prv_d in p]    # 어제 인식한 결과(csv) 목록

                    # 2시간 단위로 데이터 취합
                    hours = list(map(lambda x: int(x[11:13]), pred))
                    bins = list(range(0,25,2))
                    time_cut = pd.cat(hours, bins, right=False)
                    # (0~2: 0), (2~4: 1), (4~6: 2), ... 형식으로 시간 코드 생성/구분
                    time_code = time_cut.get_indexer(hours)
                    # 시간 코드에 따라 데이터 구분
                    df_time = pd.DataFrame({'files':pred, 'time_code':time_code})

                    #일별 데이터 취합
                    for i in range(0,12):
                        # 시간대별 데이터 취합
                        for j in df_time[df_time['time_code'==i]]['files']:
                            df = pd.read_csv(j+'/predictions.csv', header=None, names=['img_name','prediction','confidence'], index_col=1)
                            df_cnt = df.groupby('prediction')[['img_name']].count()
                            if len(df_cnt) == 0:
                                df_cnt = pd.DataFrame(index=['normal'], columns=['img_name'], data=[[0]])
                                df_cnt.index.name = 'prediction'
                            df_cnt.columns=['%02d~%02d' % (i,i+2)]
                            cnt_d = cnt_d + df_cnt
                            cnt_d.fillna(0, inplace=True)

                    cnt_d.to_csv('daily/prediction_%s.csv' % prv_d)


            # 매월 초 월별 데이터 정리
            if (now.day == 1) & (now.strftime('%H:%M')=='00:00'):
                prv_m = (now.date() - timedelta(days=1)).strftime('%Y-%m')    # 이전 달 년월(20??-??)

                # 데이터를 취합한 자료가 이미 있다면 실행하지 않음
                if 'prediction_{}'.format(prv_m) not in os.listdir('./monthly'):

                    cnt_m = pd.DataFrame(index=idx, columns=cols, data=np.zeros((6,12)))    # 데이터 취합에 사용할 객체
                    mnth = int(prv_m[-2:])    # 이전 월(month) 수

                    if mnth in [1,3,5,7,8,10,12]:    # 한 달이 31일인 경우
                        days = list(range(1,32))
                    elif mnth in [4,6,9,11]:    # 한 달이 30일인 경우
                        days = list(range(1,31))
                    else: 
                        if int(prv_m[:4]) in range(2024, 2100, 4):
                            days = list(range(1,30))    # 윤년 2월인 경우
                        else:
                            days = list(range(1,29))    # 그 외 2월인 경우

                    # 월별 데이터 취합
                    for x in days:
                        df = pd.read_csv('daily/prediction_%s-%02d.csv' % (prv_m, x), index_col=0)
                        cnt_m += df
                        cnt_m.fillna(0, inplace=True)

                    cnt_m.to_csv('monthly/prediction_%s.csv' % prv_m)

            

            
            
            
            # -------------------- 임시 이미지 분석 코드 
            
            
            pth = './yolov5/runs/detect'
            fil = [f for f in os.listdir(pth)]

            data = pd.read_csv(pth+'/'+fil[-1]+'/predictions.csv', header=0, names=['img_name','prediction','confidence'], index_col=1)
            data = data.groupby('prediction')[['img_name']].count()
            print(data)
            if len(data) == 0:
                total_sum = 0
            else:
                if 'normal' in data.index:
                    normal = data.loc['normal'].img_name
                else:
                    normal = 0
                    
                total_sum = data.sum() - normal
                total_sum = total_sum.img_name
            
            ## 기본 시간 10초
            basic = 10
            
            print('total_sum:', total_sum)
            if total_sum > 0:
                basic += 10  # basic 변수에 10을 더하여 다시 basic 변수에 할당
            print("Sending duration:", basic)
            
            # 파이썬에서 시간을 아두이노로 보냄
            ser.write(f"{basic}\n".encode())
            print("Sent duration to Arduino")
            
            # 아두이노가 주어진 시간동안 작업을 수행하도록 대기
            time.sleep(basic + 1)  # +1초 추가하여 여유시간 확보
            
    except KeyboardInterrupt:

        print("Interrupted by user")
        
    finally:
        ser.close()

# 메인 함수 실행
main()