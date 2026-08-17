#include <SPI.h>
#include <RF24.h>
#include <SoftwareSerial.h>
#include <TinyGPS++.h>

// --- ASIGNACIÓN DE PINES REALES ---
const uint8_t PIN_RF_CE = 9;   
const uint8_t PIN_RF_CSN = 10; 

const int PIN_GPS_TX = 4;   
const int PIN_GPS_RX = 6;   
const int PIN_GPS_PPS = 7;  

// --- CONFIGURACIÓN DE OBJETOS ---
RF24 radio(PIN_RF_CE, PIN_RF_CSN);
SoftwareSerial gpsSerial(PIN_GPS_TX, PIN_GPS_RX);
TinyGPSPlus gps;

bool estadoAnteriorPPS = LOW;

// Comando u-blox para forzar un Cold Start (Borrado total de caché interna)
const byte cmdColdStart[] = {
  0xB5, 0x62, 0x06, 0x04, 0x04, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x0C, 0x5D
};

void setup() {
  delay(500); 
  Serial.begin(115200);
  
  while(Serial.available() > 0) { Serial.read(); }
  Serial.println(); 

  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);
  pinMode(PIN_GPS_PPS, INPUT);

  gpsSerial.begin(9600);
  delay(200);

  // ENVIAR COMANDO DE REINICIO EN FRÍO AL CHIP GPS
  Serial.println(F("# Enviando comando Cold Start al chip GPS..."));
  gpsSerial.write(cmdColdStart, sizeof(cmdColdStart));
  delay(500);
  
  while(gpsSerial.available() > 0) { gpsSerial.read(); }

  Serial.println(F("# ----------------------------------------------------"));
  Serial.println(F("# Transceptor Activo + Fuerza de Borrado de Hardware"));
  Serial.println(F("# Radio: CE=9 CSN=10 IRQ=3 | GPS: TXD=4 RXD=6 PPS=7"));
  Serial.println(F("# ----------------------------------------------------"));

  bool ok = radio.begin();
  if (ok) {
    radio.setPALevel(RF24_PA_LOW);
    radio.setDataRate(RF24_1MBPS);
    radio.setChannel(76);
    radio.setPayloadSize(32); 
    radio.openWritingPipe((const uint8_t *)"ANE01");
    radio.openReadingPipe(1, (const uint8_t *)"ANA01");
    radio.startListening();   
    Serial.println(F("RF24_CONFIG,OK"));
  }
}

void loop() {
  // 1. GPS: Procesar tramas en segundo plano
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // 2. RADIO (RECEPCIÓN): Capturar mensajes de la red
  if (radio.available()) {
    char bufferRecibido = {0};
    radio.read(&bufferRecibido, sizeof(bufferRecibido));
    Serial.print(F("RADIO_RX,"));
    Serial.println(bufferRecibido);
  }

  // 3. REPORTES LOCALES Y TRANSMISIÓN AUTOMÁTICA
  static unsigned long last = 0;
  if (millis() - last > 1000UL) {
    last = millis();
    
    Serial.print(F("RF24_STATUS,"));
    Serial.print(radio.isChipConnected() ? 1 : 0);

    if (gps.location.isUpdated() && gps.location.isValid()) {
      Serial.print(F(",GPS_LAT,")); Serial.print(gps.location.lat(), 6);
      Serial.print(F(",GPS_LNG,")); Serial.print(gps.location.lng(), 6);
      Serial.print(F(",GPS_SAT,")); Serial.println(gps.satellites.value());
    } else {
      Serial.print(F(",GPS,Escaneando cielo de Llay Llay...[Satellites:"));
      Serial.print(gps.satellites.value());
      Serial.println(F("]"));
    }
    
    if (radio.isChipConnected()) {
      const char miMensaje[] = "DATOS_OK"; 
      radio.stopListening(); 
      radio.write(&miMensaje, sizeof(miMensaje));
      radio.startListening(); 
    }
  }

  // 4. GPS (PPS): Monitor del pulso de sincronización
  bool estadoActualPPS = digitalRead(PIN_GPS_PPS);
  if (estadoActualPPS == HIGH && estadoAnteriorPPS == LOW) {
    Serial.print(F("# PPS_PULSE_DETECTED_AT:"));
    Serial.println(millis());
  }
  estadoAnteriorPPS = estadoActualPPS;
}
