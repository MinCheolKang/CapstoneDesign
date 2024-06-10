const int greenLED2 = 11;
const int greenLED = 9;
const int redLED2 = 10;
const int redLED = 8;
int greenDuration = 0;

void setup() {
  Serial.begin(9600);
  pinMode(greenLED, OUTPUT);
  pinMode(greenLED2, OUTPUT);
  pinMode(redLED, OUTPUT);
  pinMode(redLED2, OUTPUT);
  digitalWrite(redLED, HIGH); 
  digitalWrite(redLED2, HIGH); // 시작 시 빨간 LED 켜기
}

void loop() {
  if (Serial.available() > 0) {
    greenDuration = Serial.parseInt();
    Serial.print("Received duration: ");
    Serial.println(greenDuration);
    controlGreenLED(greenDuration);
    Serial.println("green off");
  }
}

void controlGreenLED(int seconds) {
  digitalWrite(redLED, LOW);
  digitalWrite(redLED2, LOW);   // 빨간 LED 끄기
  for (int i = 0; i < seconds; i++) {
    digitalWrite(greenLED, HIGH);
    digitalWrite(greenLED2, HIGH);
    if (i >= seconds - 10) {  // 마지막 10초 동안 깜빡임
      delay(500);
      digitalWrite(greenLED, LOW);
      digitalWrite(greenLED2, LOW);
      delay(500);
    } else {
      delay(1000);
    }
  }
  digitalWrite(greenLED, LOW);
  digitalWrite(greenLED2, LOW);
  digitalWrite(redLED, HIGH);
  digitalWrite(redLED2, HIGH);    // 빨간 LED 켜기
  delay(5000);  // 빨간 LED가 켜진 상태 유지 시간
}
