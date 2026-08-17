#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>
#include <TouchScreen.h>

#define YP A3
#define XM A2
#define YM 9
#define XP 8

MCUFRIEND_kbv tft;
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

void quietTftBus() {
  pinMode(A0, OUTPUT); // LCD_RD
  pinMode(A1, OUTPUT); // LCD_WR
  pinMode(A3, OUTPUT); // LCD_CS / YP
  digitalWrite(A0, HIGH);
  digitalWrite(A1, HIGH);
  digitalWrite(A3, HIGH);
}

void restorePins() {
  quietTftBus();
  pinMode(YP, OUTPUT);
  pinMode(XM, OUTPUT);
  pinMode(XP, OUTPUT);
  pinMode(YM, OUTPUT);
  digitalWrite(YP, HIGH);
  digitalWrite(XM, HIGH);
  digitalWrite(XP, HIGH);
  digitalWrite(YM, HIGH);
}

void setup() {
  Serial.begin(9600);
  delay(500);
  uint16_t id = tft.readID();
  tft.begin(0x7793);
  tft.setRotation(1);
  tft.fillScreen(0x001F);
  tft.setTextColor(0xFFFF, 0x001F);
  tft.setTextSize(2);
  tft.setCursor(20, 40);
  tft.print("TFT + TOUCH TEST");
  Serial.print(F("# tft_touch_opensmart ready raw=0x"));
  Serial.println(id, HEX);
}

void loop() {
  quietTftBus();
  TSPoint p = ts.getPoint();
  restorePins();
  if (p.z > 0) {
    Serial.print(F("P,"));
    Serial.print(p.x);
    Serial.print(',');
    Serial.print(p.y);
    Serial.print(',');
    Serial.println(p.z);
  }
  if (p.z > 10 && p.z < 1000) {
    tft.fillCircle(300, 40, 12, 0xF800);
  }
  delay(100);
}
