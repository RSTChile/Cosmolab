#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

void setup() {
  Serial.begin(115200);
  delay(500);
  uint16_t raw = tft.readID();
  tft.begin(0x7793);
  tft.setRotation(1);

  tft.fillScreen(0x0000);
  tft.fillRect(0, 0, 480, 38, 0x18E3);
  tft.setTextColor(0xFFFF, 0x18E3);
  tft.setTextSize(2);
  tft.setCursor(12, 10);
  tft.print("TFT ST7793 SOLO - ESTABLE");

  tft.setTextColor(0xDD65, 0x0000);
  tft.setCursor(20, 70);
  tft.print("Sin radio. Sin touch.");

  tft.drawRect(20, 120, 440, 100, 0x5D9F);
  tft.fillRect(24, 124, 432, 92, 0x080B);
  tft.setTextColor(0x5EEC, 0x080B);
  tft.setCursor(40, 155);
  tft.print("Si esto se ve limpio,");
  tft.setCursor(40, 180);
  tft.print("el conflicto es SPI/radio.");

  Serial.print(F("# tft_st7793_solo_estable raw=0x"));
  Serial.print(raw, HEX);
  Serial.println(F(" used=0x7793"));
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last > 1000UL) {
    last = millis();
    Serial.println(F("TFT_SOLO,alive"));
  }
}
