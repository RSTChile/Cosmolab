#include <TouchScreen.h>

#define YP A3
#define XM A2
#define YM 9
#define XP 8

#define MINPRESSURE 10
#define MAXPRESSURE 1000

TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

void setup() {
  Serial.begin(9600);
  delay(500);
  Serial.println(F("# touch_opensmart_a3a2_9_8 ready"));
  Serial.println(F("# pins XP=8 YP=A3 XM=A2 YM=9"));
}

void loop() {
  TSPoint p = ts.getPoint();
  if (p.z > MINPRESSURE && p.z < MAXPRESSURE) {
    Serial.print(F("TOUCH,"));
    Serial.print(p.x);
    Serial.print(',');
    Serial.print(p.y);
    Serial.print(',');
    Serial.println(p.z);
    delay(100);
  }
}
