struct Combo {
  const char *name;
  uint8_t csn;
  uint8_t sck;
  uint8_t mosi;
  uint8_t miso;
};

Combo combos[] = {
  {"CSN_D4_SCK_D3_MOSI_D5_MISO_D12", 4, 3, 5, 12},
  {"CSN_D4_SCK_D3_MOSI_D12_MISO_D5", 4, 3, 12, 5},
  {"CSN_A5_SCK_D3_MOSI_D5_MISO_D12", A5, 3, 5, 12},
  {"CSN_A5_SCK_D3_MOSI_D12_MISO_D5", A5, 3, 12, 5},
  {"CSN_A4_SCK_D3_MOSI_D5_MISO_D12", A4, 3, 5, 12},
  {"CSN_A4_SCK_D3_MOSI_D12_MISO_D5", A4, 3, 12, 5},
};

uint8_t spiTransfer(const Combo &c, uint8_t out) {
  uint8_t in = 0;
  for (int i = 7; i >= 0; --i) {
    digitalWrite(c.mosi, (out >> i) & 1);
    delayMicroseconds(4);
    digitalWrite(c.sck, HIGH);
    delayMicroseconds(4);
    in <<= 1;
    if (digitalRead(c.miso)) in |= 1;
    digitalWrite(c.sck, LOW);
    delayMicroseconds(4);
  }
  return in;
}

void testCombo(const Combo &c) {
  pinMode(c.csn, OUTPUT);
  pinMode(c.sck, OUTPUT);
  pinMode(c.mosi, OUTPUT);
  pinMode(c.miso, INPUT);
  digitalWrite(c.csn, HIGH);
  digitalWrite(c.sck, LOW);
  digitalWrite(c.mosi, LOW);
  delay(5);

  digitalWrite(c.csn, LOW);
  delayMicroseconds(10);
  uint8_t status_nop = spiTransfer(c, 0xFF); // NOP returns STATUS
  digitalWrite(c.csn, HIGH);

  delay(3);
  digitalWrite(c.csn, LOW);
  delayMicroseconds(10);
  uint8_t status_read = spiTransfer(c, 0x00); // R_REGISTER CONFIG returns STATUS
  uint8_t config = spiTransfer(c, 0xFF);
  digitalWrite(c.csn, HIGH);

  Serial.print(c.name);
  Serial.print(F(",STATUS_NOP=0x"));
  if (status_nop < 16) Serial.print('0');
  Serial.print(status_nop, HEX);
  Serial.print(F(",STATUS_READ=0x"));
  if (status_read < 16) Serial.print('0');
  Serial.print(status_read, HEX);
  Serial.print(F(",CONFIG=0x"));
  if (config < 16) Serial.print('0');
  Serial.println(config, HEX);
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println(F("# nrf24_bitbang_scan"));
  for (uint8_t i = 0; i < sizeof(combos) / sizeof(combos[0]); ++i) {
    testCombo(combos[i]);
    delay(250);
  }
}

void loop() {}
