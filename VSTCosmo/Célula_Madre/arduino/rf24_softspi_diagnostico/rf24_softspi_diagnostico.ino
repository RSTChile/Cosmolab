#define SOFTSPI
#define SOFT_SPI_MISO_PIN 12
#define SOFT_SPI_MOSI_PIN 5
#define SOFT_SPI_SCK_PIN 3

#include <RF24.h>

RF24 radio_a(A5, A4); // esperado: CE=A5, CSN=A4
RF24 radio_b(A4, A5); // prueba invertida

void testRadio(const char *name, RF24 &radio) {
  bool ok = radio.begin();
  bool chip = radio.isChipConnected();
  Serial.print(name);
  Serial.print(",begin=");
  Serial.print(ok ? 1 : 0);
  Serial.print(",chip=");
  Serial.println(chip ? 1 : 0);
  if (ok && chip) {
    radio.setPALevel(RF24_PA_LOW);
    radio.setDataRate(RF24_1MBPS);
    radio.setChannel(76);
    radio.stopListening();
  }
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("# rf24_softspi_diagnostico");
  Serial.println("# SoftSPI: MISO=D12 MOSI=D5 SCK=D3");
  testRadio("CE_A5_CSN_A4", radio_a);
  delay(500);
  testRadio("CE_A4_CSN_A5", radio_b);
}

void loop() {
  delay(2000);
  testRadio("CE_A5_CSN_A4", radio_a);
  delay(500);
  testRadio("CE_A4_CSN_A5", radio_b);
}
