#include <SPI.h>
#include <RF24.h>
#include <SoftwareSerial.h>
#include <TinyGPS++.h>

// ANIMA GPS + radio digital nRF24
// Arduino Uno: GPS NEO-M8N + nRF24L01.
// Serial hacia la Pi:
//   RF24_CONFIG,OK,TXMANUAL
//   RF24_STATUS,<connected>,GPS_LAT,<lat>,GPS_LNG,<lon>,GPS_SAT,<n>,GPS_TIME,<hh:mm:ss>,GPS_DATE,<yyyy-mm-dd>
//   RADIO_RX,<payload>
//   RFTX,ok|fail,<payload>
//   # PPS_PULSE_DETECTED_AT:<millis>

const uint8_t PIN_RF_CE = 9;
const uint8_t PIN_RF_CSN = 10;

const int PIN_GPS_TX = 4;
const int PIN_GPS_RX = 6;
const int PIN_GPS_PPS = 7;

const byte RF_ADDR_SELF[6] = "ANA01";  // ANIMA escucha aqui
const byte RF_ADDR_PEER[6] = "ANE01";  // Organismo E escucha aqui

RF24 radio(PIN_RF_CE, PIN_RF_CSN);
SoftwareSerial gpsSerial(PIN_GPS_TX, PIN_GPS_RX);
TinyGPSPlus gps;

bool estadoAnteriorPPS = LOW;
uint32_t txCount = 0;
uint32_t rxCount = 0;

const byte cmdColdStart[] = {
  0xB5, 0x62, 0x06, 0x04, 0x04, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x0C, 0x5D
};

void print2(uint8_t v) {
  if (v < 10) Serial.print('0');
  Serial.print(v);
}

void printGpsTimeDate() {
  Serial.print(F(",GPS_TIME,"));
  if (gps.time.isValid()) {
    print2(gps.time.hour());
    Serial.print(':');
    print2(gps.time.minute());
    Serial.print(':');
    print2(gps.time.second());
  }

  Serial.print(F(",GPS_DATE,"));
  if (gps.date.isValid()
      && gps.date.year() >= 2020
      && gps.date.month() >= 1
      && gps.date.month() <= 12
      && gps.date.day() >= 1
      && gps.date.day() <= 31) {
    Serial.print(gps.date.year());
    Serial.print('-');
    print2(gps.date.month());
    Serial.print('-');
    print2(gps.date.day());
  }
}

bool radioTransmit(const char *payload) {
  char msg[32] = {0};
  strncpy(msg, payload, sizeof(msg) - 1);

  radio.stopListening();
  bool ok = radio.write(&msg, sizeof(msg));
  radio.startListening();

  if (ok) txCount++;
  Serial.print(F("RFTX,"));
  Serial.print(ok ? F("ok,") : F("fail,"));
  Serial.println(msg);
  return ok;
}

void procesaSerialPi() {
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '\r' || cmd == '\n') {
      continue;
    }
    if (cmd == 'T') {
      char msg[32] = {0};
      Serial.readBytesUntil('\n', msg, sizeof(msg) - 1);
      if (msg[0] != '\0') {
        radioTransmit(msg);
      }
    } else {
      while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n') break;
      }
    }
  }
}

void setup() {
  delay(500);
  Serial.begin(115200);
  Serial.setTimeout(60);

  while (Serial.available() > 0) Serial.read();
  Serial.println();

  pinMode(PIN_RF_CSN, OUTPUT);
  digitalWrite(PIN_RF_CSN, HIGH);
  pinMode(PIN_GPS_PPS, INPUT);

  gpsSerial.begin(9600);
  delay(200);

  Serial.println(F("# Enviando comando Cold Start al chip GPS..."));
  gpsSerial.write(cmdColdStart, sizeof(cmdColdStart));
  delay(500);
  while (gpsSerial.available() > 0) gpsSerial.read();

  Serial.println(F("# ----------------------------------------------------"));
  Serial.println(F("# ANIMA GPS+nRF24 organelo"));
  Serial.println(F("# Radio: CE=9 CSN=10 | GPS: TXD=4 RXD=6 PPS=7"));
  Serial.println(F("# ----------------------------------------------------"));

  bool ok = radio.begin();
  if (ok) {
    radio.setPALevel(RF24_PA_LOW);
    radio.setDataRate(RF24_1MBPS);
    radio.setChannel(76);
    radio.setPayloadSize(32);
    radio.openWritingPipe(RF_ADDR_PEER);
    radio.openReadingPipe(1, RF_ADDR_SELF);
    radio.startListening();
    Serial.println(F("RF24_CONFIG,OK,TXMANUAL"));
  } else {
    Serial.println(F("RF24_CONFIG,FAIL"));
  }
}

void loop() {
  procesaSerialPi();

  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  if (radio.available()) {
    char bufferRecibido[32] = {0};
    radio.read(&bufferRecibido, sizeof(bufferRecibido));
    rxCount++;
    Serial.print(F("RADIO_RX,"));
    Serial.println(bufferRecibido);
  }

  static unsigned long last = 0;
  if (millis() - last > 1000UL) {
    last = millis();

    Serial.print(F("RF24_STATUS,"));
    Serial.print(radio.isChipConnected() ? 1 : 0);

    if (gps.location.isUpdated() && gps.location.isValid()) {
      Serial.print(F(",GPS_LAT,"));
      Serial.print(gps.location.lat(), 6);
      Serial.print(F(",GPS_LNG,"));
      Serial.print(gps.location.lng(), 6);
      Serial.print(F(",GPS_SAT,"));
      Serial.print(gps.satellites.value());
      printGpsTimeDate();
      Serial.println();
    } else {
      Serial.print(F(",GPS,Escaneando cielo de Llay Llay...[Satellites:"));
      Serial.print(gps.satellites.value());
      Serial.println(F("]"));
    }

    if (radio.isChipConnected()) {
      radioTransmit("DATOS_OK");
    }
  }

  bool estadoActualPPS = digitalRead(PIN_GPS_PPS);
  if (estadoActualPPS == HIGH && estadoAnteriorPPS == LOW) {
    Serial.print(F("# PPS_PULSE_DETECTED_AT:"));
    Serial.println(millis());
  }
  estadoAnteriorPPS = estadoActualPPS;
}
