#include <TouchScreen.h>

TouchScreen ts0(6, A1, A2, 7, 300); // diagnose direct
TouchScreen ts1(7, A2, A1, 6, 300); // diagnose reversed
TouchScreen ts2(6, A2, A1, 7, 300);
TouchScreen ts3(7, A1, A2, 6, 300);

void restoreAll() {
  pinMode(A1, OUTPUT);
  pinMode(A2, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);
}

void printPoint(const char *name, TSPoint p) {
  Serial.print(name);
  Serial.print(',');
  Serial.print(p.x);
  Serial.print(',');
  Serial.print(p.y);
  Serial.print(',');
  Serial.println(p.z);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("# touch_combo_diagnostico ready"));
  Serial.println(F("# press and hold; looking for z changing above zero"));
}

void loop() {
  TSPoint p0 = ts0.getPoint();
  restoreAll();
  TSPoint p1 = ts1.getPoint();
  restoreAll();
  TSPoint p2 = ts2.getPoint();
  restoreAll();
  TSPoint p3 = ts3.getPoint();
  restoreAll();

  printPoint("T0_XP6_YPA1_XMA2_YM7", p0);
  printPoint("T1_XP7_YPA2_XMA1_YM6", p1);
  printPoint("T2_XP6_YPA2_XMA1_YM7", p2);
  printPoint("T3_XP7_YPA1_XMA2_YM6", p3);
  Serial.println(F("--"));
  delay(450);
}
