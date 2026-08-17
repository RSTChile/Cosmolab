#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

const uint16_t IDS[] = {
  0x7793,
  0x0808,
  0x9486,
  0x9488,
  0x9481,
  0x9327,
  0x8357,
  0x9090,
  0x9341,
  0x7796,
  0x1580,
  0x1963
};

const uint16_t COLORS[] = {
  0xF800, 0x07E0, 0x001F, 0xFFE0, 0x07FF, 0xF81F, 0xFFFF
};

void drawTest(uint16_t id, uint8_t n) {
  tft.begin(id);
  tft.setRotation(1);
  tft.fillScreen(COLORS[n % (sizeof(COLORS) / sizeof(COLORS[0]))]);
  delay(180);
  tft.fillRect(0, 0, tft.width(), 42, 0x0000);
  tft.setTextSize(2);
  tft.setTextColor(0xFFFF, 0x0000);
  tft.setCursor(8, 10);
  tft.print(F("TFT TEST ID 0x"));
  tft.print(id, HEX);

  tft.setTextColor(0x0000, 0xFFFF);
  tft.fillRect(16, 62, tft.width() - 32, 52, 0xFFFF);
  tft.setCursor(28, 78);
  tft.print(F("ANIMA RADIO"));

  tft.drawRect(16, 132, tft.width() - 32, 72, 0x0000);
  tft.drawLine(16, 132, tft.width() - 16, 204, 0x0000);
  tft.drawLine(tft.width() - 16, 132, 16, 204, 0x0000);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  uint16_t raw = tft.readID();
  Serial.print(F("# TFT raw readID=0x"));
  Serial.println(raw, HEX);
  Serial.println(F("# Cycling TFT controller IDs every 3 seconds"));
}

void loop() {
  static uint8_t i = 0;
  uint16_t id = IDS[i % (sizeof(IDS) / sizeof(IDS[0]))];
  Serial.print(F("TRY_ID,0x"));
  Serial.println(id, HEX);
  drawTest(id, i);
  i++;
  delay(3000);
}
