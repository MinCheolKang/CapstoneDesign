import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import serial, random, time
import os, shlex, subprocess
import requests, cv2

# ESP32CAM�� IP �ּҿ� ��Ʈ
url = 'http://192.168.60.85/'

# �� �ν�/�м� �𵨰� �̹��� resize ũ�� ����
loaded_model = tf.keras.models.load_model('image_classification_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128,128

# ��Ʈ���� ���� ��û�� ������ �Լ�
def start_stream():
    response = requests.get(url + 'control?var=start')
    if response.status_code == 200:
        print("Streaming started successfully")
    else:
        print("Failed to start streaming")

# ���� �ð��� ���� �̸����� �ϴ� �̹����� �����ϴ� �Լ�
def save_image():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response = requests.get(url + 'capture', stream=True)
    if response.status_code == 200:
        with open(f"{timestamp}.jpg", 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {timestamp}.jpg")
    else:
        print("Failed to save image")

# �� �ν�, ���� �з� �Լ�
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

# ���� �Լ�
def main():
    ser = serial.Serial('COM5', 9600, timeout=1)  # ��Ʈ �̸��� �ý��ۿ� ���� �ٸ� �� ����
    time.sleep(2)  # �ø��� ��Ʈ �ʱ�ȭ ���
    global flag
    flag = None
    
    try:
        while True:
            # ���� LED�� ���� �����̶�� ��ȣ�� �Ƶ��̳뿡�� ����
            ser.write(b"Turn off red LED\n")
            print("Sent signal to turn off red LED")
            
            # �Ƶ��̳�κ��� �۾� �Ϸ� ��ȣ�� ��ٸ�
            response = ser.readline().decode('utf-8').strip()
            if response == "Done":
                print("Arduino has finished its job")
            
            time.sleep(1)  # �۾� �Ϸ� ��ȣ�� ���� �� ��� ���

            #---------------------------------���� �Կ�---------------------------------------



            # 'Start Stream' ��ư Ŭ��
            start_stream()
            
            save_image()
            
            # ��Ʈ���� ���� ��û
            response = requests.get(url + 'control?var=stop')
            if response.status_code == 200:
                print("Streaming stopped successfully")
            else:
                print("Failed to stop streaming")

            


           
            #---------------------------#yolov5 �̹��� �м�---------------------------------


            
            pic = [p for p in os.listdir('.') if '.jpg' in p]
            if pic[-1] == flag:
                pass
            else:
                command = shlex.split('python yolov5/detect.py --save-csv --weights best.pt --conf 0.5 --source '+str(pic[-1]))
                subprocess.run(command)
                flag = pic[-1]


            
            
            # ------------------- ---------����, ���� �м�--------- -------------------- 


            # �̹��� ���� �� ��� ����
            image_path = str(pic[-1])
            results = detect_and_classify_faces(image_path)
            
            # ����� CSV�� ����
            results_df = pd.DataFrame(results)

            folder = [f for f in os.listdir('yolov5/runs/detect')][-1]
            chk = os.listdir('yolov5/runs/detect/'+folder)

            # ī�޶� �� � ����� ������ ���� ��� csv������ �������� ����. �̸� ���� ó��
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

            # ����,�Ϻ� ������ ���� ���� ����
            if 'monthly' not in os.listdir('.'):
                os.mkdir('monthly')
            if 'daily' not in os.listdir('.'):
                os.mkdir('daily')


            now = datetime.now()

            # ���� ���� �Ϻ� ������ ����
            if (now.strftime('%H:%M:%S')=='00:00:00') & (now.microsecond==0):
                prv_d = (now.date() - timedelta(days=1)).strftime('%Y-%m-%d')    # ���� ��¥(20??-??-??)

                 # �����͸� ������ �ڷᰡ �̹� �ִٸ� �������� ����
                if 'prediction_{}'.format(prv_d) not in os.listdir('./daily'):

                    cnt_d = pd.DataFrame(index=idx, columns=cols, data=np.zeros((6,12)))    # ������ ���տ� ����� ��ü
                    pred = [p for p in os.listdir('.') if prv_d in p]    # ���� �ν��� ���(csv) ���

                    # 2�ð� ������ ������ ����
                    hours = list(map(lambda x: int(x[11:13]), pred))
                    bins = list(range(0,25,2))
                    time_cut = pd.cat(hours, bins, right=False)
                    # (0~2: 0), (2~4: 1), (4~6: 2), ... �������� �ð� �ڵ� ����/����
                    time_code = time_cut.get_indexer(hours)
                    # �ð� �ڵ忡 ���� ������ ����
                    df_time = pd.DataFrame({'files':pred, 'time_code':time_code})

                    #�Ϻ� ������ ����
                    for i in range(0,12):
                        # �ð��뺰 ������ ����
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


            # �ſ� �� ���� ������ ����
            if (now.day == 1) & (now.strftime('%H:%M')=='00:00'):
                prv_m = (now.date() - timedelta(days=1)).strftime('%Y-%m')    # ���� �� ���(20??-??)

                # �����͸� ������ �ڷᰡ �̹� �ִٸ� �������� ����
                if 'prediction_{}'.format(prv_m) not in os.listdir('./monthly'):

                    cnt_m = pd.DataFrame(index=idx, columns=cols, data=np.zeros((6,12)))    # ������ ���տ� ����� ��ü
                    mnth = int(prv_m[-2:])    # ���� ��(month) ��

                    if mnth in [1,3,5,7,8,10,12]:    # �� ���� 31���� ���
                        days = list(range(1,32))
                    elif mnth in [4,6,9,11]:    # �� ���� 30���� ���
                        days = list(range(1,31))
                    else: 
                        if int(prv_m[:4]) in range(2024, 2100, 4):
                            days = list(range(1,30))    # ���� 2���� ���
                        else:
                            days = list(range(1,29))    # �� �� 2���� ���

                    # ���� ������ ����
                    for x in days:
                        df = pd.read_csv('daily/prediction_%s-%02d.csv' % (prv_m, x), index_col=0)
                        cnt_m += df
                        cnt_m.fillna(0, inplace=True)

                    cnt_m.to_csv('monthly/prediction_%s.csv' % prv_m)

            

            
            
            
            # -------------------- �ӽ� �̹��� �м� �ڵ� 
            
            
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
            
            ## �⺻ �ð� 10��
            basic = 10
            
            print('total_sum:', total_sum)
            if total_sum > 0:
                basic += 10  # basic ������ 10�� ���Ͽ� �ٽ� basic ������ �Ҵ�
            print("Sending duration:", basic)
            
            # ���̽㿡�� �ð��� �Ƶ��̳�� ����
            ser.write(f"{basic}\n".encode())
            print("Sent duration to Arduino")
            
            # �Ƶ��̳밡 �־��� �ð����� �۾��� �����ϵ��� ���
            time.sleep(basic + 1)  # +1�� �߰��Ͽ� �����ð� Ȯ��
            
    except KeyboardInterrupt:

        print("Interrupted by user")
        
    finally:
        ser.close()

# ���� �Լ� ����
main()