#include <RF24.h>

RF24 radio_a(A4, A5); // esperado: CE=A4 CSN=A5
RF24 radio_b(A5, A4); // prueba: CE=A5 CSN=A4

void testRadio(const char *label, RF24 &r) {
  bool ok = r.begin();
  Serial.print(label);
  Serial.print(F(",begin="));
  Serial.print(ok ? 1 : 0);
  Serial.print(F(",connected="));
  Serial.println(r.isChipConnected() ? 1 : 0);
  r.powerDown();
  delay(200);
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println(F("# rf24_softspi_scan_ce_csn"));
  Serial.println(F("# SOFTSPI expected SCK=D3 MOSI=D5 MISO=D12"));
  testRadio("A_CE_A4_CSN_A5", radio_a);
  testRadio("B_CE_A5_CSN_A4", radio_b);
}

void loop() {
  delay(1000);
}
