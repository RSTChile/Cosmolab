#include <TouchScreen.h>

// Detectado por MCUFRIEND diagnose_Touchpins:
//   YP,YM = A1,D7   XM,XP = A2,D6
const int XP = 6;
const int YP = A1;
const int XM = A2;
const int YM = 7;

TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("# touch_raw_diagnostico ready"));
  Serial.println(F("# pins XP=6 YP=A1 XM=A2 YM=7"));
}

void loop() {
  TSPoint p = ts.getPoint();
  pinMode(YP, OUTPUT);
  pinMode(XM, OUTPUT);
  if (p.z > 0) {
    Serial.print(F("TOUCHRAW,"));
    Serial.print(p.x);
    Serial.print(',');
    Serial.print(p.y);
    Serial.print(',');
    Serial.println(p.z);
  }
  delay(80);
}
