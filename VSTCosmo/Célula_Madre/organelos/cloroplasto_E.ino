/*
 * cloroplasto_E.ino  —  Sensor físico de E (panel + LiPo + GPS/PPS)
 * ================================================================
 * ATmega 2560 Pro + CH340G → USB-serie → Raspberry Pi @ 115200
 * Anti-Shannon: solo magnitudes crudas; la Pi interpreta.
 *
 * CABLEADO:
 *   A0  <- panel/fuente (directo, max 5V; con divisor real usar DIV_A0)
 *   A1  <- LiPo 3,7V directo
 *   RX2 <- GPS TX (NEO-M8N @ 9600)
 *   D2  <- GPS PPS
 *   D3  <- nRF24 IRQ
 *   D4  -> RD-WiFi EN/CH_PD (opcional; D2 queda reservado al PPS)
 *   D18/TX1 -> RD-WiFi RXD
 *   D19/RX1 <- RD-WiFi TXD
 *   GND <- tierra común del subsistema solar
 *
 * SALIDA (1 Hz):
 *   LUZ,<v_fuente_V>,<v_lipo_V>,<adc_A0>,<adc_A1>
 *   GPS,<fix>,<sats>,<hdop>,<lat>,<lon>,<alt_m>,<speed_kn>,<course_deg>,<pps_count>,<pps_age_ms>,<nmea_seen>
 */

#include <SPI.h>
#include <RF24.h>
#include "wifi_config.h"

// --- Radio digital nRF24L01 (2.4 GHz ISM, paquetes) — la 2a radio de E (endogena, en la ATmega) ---
RF24 radio(9, 10);                 // CE=D9, CSN=D10 (SPI hw: D50 MISO, D51 MOSI, D52 SCK)
const int   PIN_RF_IRQ = 3;        // IRQ del nRF24 (D2 lo tiene el GPS PPS)
const byte  RF_ADDR_SELF[6] = "ANE01";  // direccion de E (escucha aqui)
const byte  RF_ADDR_PEER[6] = "ANA01";  // direccion del par (transmite alla)
bool  rf_ok = false;
volatile bool rf_irq = false;
unsigned long rf_rx_count = 0, rf_tx_count = 0;
char rf_cmd[40]; uint8_t rf_cmd_len = 0;
void on_rf_irq() { rf_irq = true; }

const int   PIN_PANEL = A0;
const int   PIN_LIPO  = A1;
const int   PIN_PPS   = 2;
const int   PIN_WIFI_EN = 4;
const float VREF      = 5.0;
const float DIV_A0    = 1.0;    // 0.5 si hay divisor 2:1 en panel
const float DIV_A1    = 1.0;

const int   N_PROM    = 16;
const unsigned long PERIODO_MS = 1000UL;

unsigned long t_prev = 0;
volatile unsigned long pps_count = 0;
volatile unsigned long pps_last_ms = 0;

char gps_line[128];
uint8_t gps_line_len = 0;
unsigned long nmea_seen = 0;
bool gps_fix = false;
int gps_sats = 0;
float gps_hdop = 0.0;
float gps_lat = 0.0;
float gps_lon = 0.0;
float gps_alt_m = 0.0;
float gps_speed_kn = 0.0;
float gps_course_deg = 0.0;

bool wifi_ready = false;
bool wifi_disabled = false;
unsigned long wifi_last_send_ms = 0;
unsigned long wifi_baud_actual = WIFI_BAUD;

void on_pps() {
  pps_count++;
  pps_last_ms = millis();
}

int leer_promediado(int pin) {
  long acc = 0;
  for (int i = 0; i < N_PROM; i++) {
    acc += analogRead(pin);
    delay(2);
  }
  return (int)(acc / N_PROM);
}

float nmea_coord_to_decimal(const char *coord, const char *hemi) {
  if (!coord || !coord[0]) return 0.0;
  float raw = atof(coord);
  int deg = (int)(raw / 100.0);
  float minutes = raw - (deg * 100.0);
  float dec = deg + minutes / 60.0;
  if (hemi && (hemi[0] == 'S' || hemi[0] == 'W')) dec = -dec;
  return dec;
}

void parse_nmea(char *line) {
  if (line[0] != '$') return;
  nmea_seen++;
  char *star = strchr(line, '*');
  if (star) *star = '\0';
  char *fields[20];
  int n = 0;
  char *save = NULL;
  char *tok = strtok_r(line, ",", &save);
  while (tok && n < 20) {
    fields[n++] = tok;
    tok = strtok_r(NULL, ",", &save);
  }
  if (n == 0) return;
  if (strstr(fields[0], "RMC") && n >= 9) {
    gps_fix = (fields[2][0] == 'A');
    if (fields[3][0] && fields[5][0]) {
      gps_lat = nmea_coord_to_decimal(fields[3], fields[4]);
      gps_lon = nmea_coord_to_decimal(fields[5], fields[6]);
    }
    gps_speed_kn = fields[7][0] ? atof(fields[7]) : 0.0;
    gps_course_deg = fields[8][0] ? atof(fields[8]) : 0.0;
  } else if (strstr(fields[0], "GGA") && n >= 10) {
    int fix_quality = fields[6][0] ? atoi(fields[6]) : 0;
    gps_fix = fix_quality > 0;
    gps_sats = fields[7][0] ? atoi(fields[7]) : 0;
    gps_hdop = fields[8][0] ? atof(fields[8]) : 0.0;
    gps_alt_m = fields[9][0] ? atof(fields[9]) : 0.0;
    if (fields[2][0] && fields[4][0]) {
      gps_lat = nmea_coord_to_decimal(fields[2], fields[3]);
      gps_lon = nmea_coord_to_decimal(fields[4], fields[5]);
    }
  }
}

void leer_gps() {
  while (Serial2.available()) {
    char c = (char)Serial2.read();
    if (c == '\r') continue;
    if (c == '\n') {
      gps_line[gps_line_len] = '\0';
      parse_nmea(gps_line);
      gps_line_len = 0;
    } else if (gps_line_len < sizeof(gps_line) - 1) {
      gps_line[gps_line_len++] = c;
    } else {
      gps_line_len = 0;
    }
  }
}

void wifi_drain(unsigned long ms = 20) {
#if WIFI_ENABLED
  unsigned long start = millis();
  while (millis() - start < ms) {
    while (Serial1.available()) Serial1.read();
  }
#endif
}

bool wifi_wait_for(const char *needle, unsigned long timeout_ms) {
#if WIFI_ENABLED
  char window[96];
  uint8_t len = 0;
  window[0] = '\0';
  unsigned long start = millis();
  while (millis() - start < timeout_ms) {
    while (Serial1.available()) {
      char c = (char)Serial1.read();
      if (len < sizeof(window) - 1) {
        window[len++] = c;
      } else {
        memmove(window, window + 1, sizeof(window) - 2);
        window[sizeof(window) - 2] = c;
        len = sizeof(window) - 1;
      }
      window[len] = '\0';
      if (strstr(window, needle)) return true;
      if (strstr(window, "ERROR") || strstr(window, "FAIL")) return false;
    }
  }
#else
  (void)needle;
  (void)timeout_ms;
#endif
  return false;
}

bool wifi_cmd(const char *cmd, const char *expect, unsigned long timeout_ms) {
#if WIFI_ENABLED
  wifi_drain();
  Serial1.println(cmd);
  return wifi_wait_for(expect, timeout_ms);
#else
  (void)cmd;
  (void)expect;
  (void)timeout_ms;
  return false;
#endif
}

bool wifi_connect() {
#if WIFI_ENABLED
  char cmd[120];
  if (wifi_disabled) return false;
  wifi_ready = false;

  pinMode(PIN_WIFI_EN, OUTPUT);
  digitalWrite(PIN_WIFI_EN, LOW);
  delay(400);
  digitalWrite(PIN_WIFI_EN, HIGH);
  delay(3000);

  const unsigned long bauds[] = {WIFI_BAUD, 9600UL, 57600UL, 38400UL};
  bool at_ok = false;
  for (uint8_t i = 0; i < sizeof(bauds) / sizeof(bauds[0]); i++) {
    wifi_baud_actual = bauds[i];
    Serial1.end();
    delay(80);
    Serial1.begin(wifi_baud_actual);
    if (wifi_cmd("AT", "OK", 1800) || wifi_cmd("AT", "OK", 1800)) {
      at_ok = true;
      Serial.print("# WIFI: AT OK baud=");
      Serial.println(wifi_baud_actual);
      break;
    }
  }

  if (!at_ok) {
    Serial.println("# WIFI: no responde AT");
    wifi_disabled = true;
    digitalWrite(PIN_WIFI_EN, LOW);
    return false;
  }
  wifi_cmd("ATE0", "OK", 1000);
  wifi_cmd("AT+CWMODE=1", "OK", 1500);
  wifi_cmd("AT+CIPMUX=0", "OK", 1500);

  snprintf(cmd, sizeof(cmd), "AT+CWJAP_CUR=\"%s\",\"%s\"", WIFI_SSID, WIFI_PASS);
  if (!wifi_cmd(cmd, "OK", 15000)) {
    Serial.println("# WIFI: no conecto a la red");
    wifi_disabled = true;
    digitalWrite(PIN_WIFI_EN, LOW);
    return false;
  }

  wifi_ready = true;
  Serial.println("# WIFI: conectado");
  return true;
#else
  pinMode(PIN_WIFI_EN, INPUT);
  return false;
#endif
}

bool wifi_send_payload(const char *payload) {
#if WIFI_ENABLED
  char cmd[96];
  snprintf(cmd, sizeof(cmd), "AT+CIPSTART=\"TCP\",\"%s\",%u", WIFI_HOST, WIFI_PORT);
  if (!wifi_cmd(cmd, "OK", 5000)) {
    wifi_ready = false;
    return false;
  }

  snprintf(cmd, sizeof(cmd), "AT+CIPSEND=%u", (unsigned int)strlen(payload));
  wifi_drain();
  Serial1.println(cmd);
  if (!wifi_wait_for(">", 3000)) {
    wifi_cmd("AT+CIPCLOSE", "OK", 1000);
    return false;
  }

  Serial1.print(payload);
  bool ok = wifi_wait_for("SEND OK", 5000);
  wifi_cmd("AT+CIPCLOSE", "OK", 1000);
  if (!ok) wifi_ready = false;
  return ok;
#else
  (void)payload;
  return false;
#endif
}

void wifi_tick(float v_fuente, float v_lipo, int cA0, int cA1,
               unsigned long pps_snapshot, unsigned long pps_age) {
#if WIFI_ENABLED
  if (wifi_disabled) return;
  unsigned long ahora = millis();
  if (ahora - wifi_last_send_ms < WIFI_INTERVAL_MS) return;
  wifi_last_send_ms = ahora;

  if (!wifi_ready && !wifi_connect()) return;

  char vf[12], vl[12], hdop[12], lat[16], lon[16], alt[12], spd[12], crs[12];
  char payload[260];
  dtostrf(v_fuente, 1, 3, vf);
  dtostrf(v_lipo, 1, 3, vl);
  dtostrf(gps_hdop, 1, 2, hdop);
  dtostrf(gps_lat, 1, 6, lat);
  dtostrf(gps_lon, 1, 6, lon);
  dtostrf(gps_alt_m, 1, 1, alt);
  dtostrf(gps_speed_kn, 1, 2, spd);
  dtostrf(gps_course_deg, 1, 2, crs);

  snprintf(payload, sizeof(payload),
           "E,%lu,LUZ,%s,%s,%d,%d,GPS,%d,%d,%s,%s,%s,%s,%s,%s,%lu,%lu,%lu,RF24,%d,%d,%lu,%lu\n",
           ahora, vf, vl, cA0, cA1,
           gps_fix ? 1 : 0, gps_sats, hdop, lat, lon, alt, spd, crs,
           pps_snapshot, pps_age, nmea_seen,
           rf_ok ? 1 : 0, radio.isChipConnected() ? 1 : 0, rf_rx_count, rf_tx_count);

  Serial.print("# WIFI TX: ");
  Serial.println(wifi_send_payload(payload) ? "ok" : "fail");
#else
  (void)v_fuente;
  (void)v_lipo;
  (void)cA0;
  (void)cA1;
  (void)pps_snapshot;
  (void)pps_age;
#endif
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600);
  analogReference(DEFAULT);
  pinMode(PIN_PPS, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_PPS), on_pps, RISING);
  for (int i = 0; i < 8; i++) { analogRead(PIN_PANEL); analogRead(PIN_LIPO); delay(5); }
  // --- init de la radio digital (2a radio) ---
  rf_ok = radio.begin();
  if (rf_ok) {
    radio.setPALevel(RF24_PA_LOW);
    radio.setDataRate(RF24_1MBPS);
    radio.setChannel(76);
    radio.openWritingPipe(RF_ADDR_PEER);
    radio.openReadingPipe(1, RF_ADDR_SELF);
    radio.startListening();
    pinMode(PIN_RF_IRQ, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_RF_IRQ), on_rf_irq, FALLING);
  }
  Serial.print("# RF24: "); Serial.print(rf_ok ? "OK" : "FAIL");
  Serial.print(" connected="); Serial.println(radio.isChipConnected() ? 1 : 0);
  wifi_connect();
  Serial.println("# cloroplasto_E ready");
}

void loop() {

  // --- radio digital: RX (reporta paquetes) + TX por comando serial "T<texto>" ---
  if (rf_ok) {
    while (radio.available()) {
      char buf[33]; buf[32]=0; radio.read(&buf, 32); rf_rx_count++;
      Serial.print("RFRX,"); Serial.println(buf);
    }
    while (Serial.available()) {
      char c=(char)Serial.read();
      if (c=='\n'||c=='\r') {
        if (rf_cmd_len>0 && rf_cmd[0]=='T') {
          rf_cmd[rf_cmd_len]=0; radio.stopListening();
          bool ok=radio.write(&rf_cmd[1], rf_cmd_len); radio.startListening();
          rf_tx_count++; Serial.print("RFTX,"); Serial.println(ok?"ok":"fail");
        } else if (rf_cmd_len>0 && rf_cmd[0]=='W') {
          Serial.println("# WIFI: reinicio manual por EN/D4");
          wifi_disabled = false;
          wifi_connect();
        } else if (rf_cmd_len>0 && rf_cmd[0]=='w') {
          Serial.println("# WIFI: apagado manual por EN/D4");
          wifi_ready = false;
          wifi_disabled = true;
          pinMode(PIN_WIFI_EN, OUTPUT);
          digitalWrite(PIN_WIFI_EN, LOW);
        }
        rf_cmd_len=0;
      } else if (rf_cmd_len<sizeof(rf_cmd)-1) { rf_cmd[rf_cmd_len++]=c; } else { rf_cmd_len=0; }
    }
  }
  leer_gps();
  unsigned long ahora = millis();
  if (ahora - t_prev < PERIODO_MS) return;
  t_prev = ahora;

  int cA0 = leer_promediado(PIN_PANEL);
  int cA1 = leer_promediado(PIN_LIPO);
  float v_fuente = (cA0 * VREF / 1023.0) / DIV_A0;
  float v_lipo   = (cA1 * VREF / 1023.0) / DIV_A1;

  Serial.print("LUZ,");
  Serial.print(v_fuente, 3); Serial.print(",");
  Serial.print(v_lipo, 3);  Serial.print(",");
  Serial.print(cA0);        Serial.print(",");
  Serial.println(cA1);

  unsigned long pps_snapshot, pps_last_snapshot;
  noInterrupts();
  pps_snapshot = pps_count;
  pps_last_snapshot = pps_last_ms;
  interrupts();
  unsigned long pps_age = pps_last_snapshot ? (millis() - pps_last_snapshot) : 999999UL;

  Serial.print("GPS,");
  Serial.print(gps_fix ? 1 : 0);    Serial.print(",");
  Serial.print(gps_sats);           Serial.print(",");
  Serial.print(gps_hdop, 2);        Serial.print(",");
  Serial.print(gps_lat, 6);         Serial.print(",");
  Serial.print(gps_lon, 6);         Serial.print(",");
  Serial.print(gps_alt_m, 1);       Serial.print(",");
  Serial.print(gps_speed_kn, 2);    Serial.print(",");
  Serial.print(gps_course_deg, 2);  Serial.print(",");
  Serial.print(pps_snapshot);       Serial.print(",");
  Serial.print(pps_age);            Serial.print(",");
  Serial.println(nmea_seen);

  Serial.print("RF24,");
  Serial.print(rf_ok?1:0); Serial.print(",");
  Serial.print(radio.isChipConnected()?1:0); Serial.print(",");
  Serial.print(rf_rx_count); Serial.print(",");
  Serial.println(rf_tx_count);

  wifi_tick(v_fuente, v_lipo, cA0, cA1, pps_snapshot, pps_age);
}
