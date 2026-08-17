#include <SPI.h>
#include <RF24.h>

// Puerto azul Open-Smart:
// GND, VCC, MISO, MOSI, SCK, D5
// Prueba principal: D5 como CSN y A5 como CE.
RF24 radio_csn_d5(A5, 5);

// Prueba alternativa: por si el adaptador externo espera D5 como CE.
RF24 radio_ce_d5(5, A5);

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
  pinMode(10, OUTPUT); // mantiene el ATmega328P como SPI master aunque CSN no sea D10
  digitalWrite(10, HIGH);
  Serial.println("# rf24_hardware_spi_diagnostico");
  Serial.println("# SPI hardware: MISO=D12 MOSI=D11 SCK=D13, puerto azul; D5=control");
  testRadio("CE_A5_CSN_D5", radio_csn_d5);
  delay(500);
  testRadio("CE_D5_CSN_A5", radio_ce_d5);
}

void loop() {
  delay(2000);
  testRadio("CE_A5_CSN_D5", radio_csn_d5);
  delay(500);
  testRadio("CE_D5_CSN_A5", radio_ce_d5);
}
