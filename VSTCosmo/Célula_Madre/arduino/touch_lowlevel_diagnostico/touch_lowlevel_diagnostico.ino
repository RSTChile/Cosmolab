const int XP = 6;
const int YP = A1;
const int XM = A2;
const int YM = 7;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("# touch_lowlevel_diagnostico ready"));
  Serial.println(F("# pins XP=6 YP=A1 XM=A2 YM=7"));
}

int readMedian(int pin) {
  int a = analogRead(pin);
  int b = analogRead(pin);
  return (a + b) / 2;
}

void loop() {
  pinMode(YP, INPUT);
  pinMode(YM, INPUT);
  pinMode(XP, OUTPUT);
  pinMode(XM, OUTPUT);
  digitalWrite(XP, HIGH);
  digitalWrite(XM, LOW);
  delayMicroseconds(20);
  int xraw = 1023 - readMedian(YP);

  pinMode(XP, INPUT);
  pinMode(XM, INPUT);
  pinMode(YP, OUTPUT);
  pinMode(YM, OUTPUT);
  digitalWrite(YM, LOW);
  digitalWrite(YP, HIGH);
  delayMicroseconds(20);
  int yraw = 1023 - readMedian(XM);

  pinMode(XP, OUTPUT);
  pinMode(YP, INPUT);
  pinMode(XM, INPUT);
  pinMode(YM, OUTPUT);
  digitalWrite(XP, LOW);
  digitalWrite(YM, HIGH);
  delayMicroseconds(20);
  int z1 = analogRead(XM);
  int z2 = analogRead(YP);
  int zcalc = 1023 - (z2 - z1);

  Serial.print(F("LOW,"));
  Serial.print(xraw);
  Serial.print(',');
  Serial.print(yraw);
  Serial.print(',');
  Serial.print(z1);
  Serial.print(',');
  Serial.print(z2);
  Serial.print(',');
  Serial.println(zcalc);
  delay(250);
}
