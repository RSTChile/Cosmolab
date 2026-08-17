#include <SPI.h>
#include <RF24.h>

const uint8_t PIN_RF_CE = A5;
const uint8_t PIN_RF_CSN = 5;

RF24 radio(PIN_RF_CE, PIN_RF_CSN);

void setup() {
  Serial.begin(115200);
  delay(700);
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);
  Serial.println(F("# rf24_hardware_spi_original"));
  Serial.println(F("# CE=A5 CSN=D5 SPI=puerto azul"));

  bool ok = radio.begin();
  bool connected = radio.isChipConnected();
  Serial.print(F("RF24_BEGIN,"));
  Serial.println(ok ? 1 : 0);
  Serial.print(F("RF24_CONNECTED,"));
  Serial.println(connected ? 1 : 0);

  if (ok) {
    radio.setPALevel(RF24_PA_LOW);
    radio.setDataRate(RF24_1MBPS);
    radio.setChannel(76);
    radio.setPayloadSize(32);
    radio.openWritingPipe((const uint8_t *)"ANE01");
    radio.openReadingPipe(1, (const uint8_t *)"ANA01");
    radio.startListening();
  }
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last > 1000UL) {
    last = millis();
    Serial.print(F("RF24_STATUS,"));
    Serial.println(radio.isChipConnected() ? 1 : 0);
  }
}
