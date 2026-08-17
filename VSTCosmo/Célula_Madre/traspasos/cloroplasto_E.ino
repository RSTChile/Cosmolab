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
 *   GND <- tierra común del subsistema solar
 *
 * SALIDA (1 Hz):
 *   LUZ,<v_fuente_V>,<v_lipo_V>,<adc_A0>,<adc_A1>
 *   GPS,<fix>,<sats>,<hdop>,<lat>,<lon>,<alt_m>,<speed_kn>,<course_deg>,<pps_count>,<pps_age_ms>,<nmea_seen>
 */

const int   PIN_PANEL = A0;
const int   PIN_LIPO  = A1;
const int   PIN_PPS   = 2;
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

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600);
  analogReference(DEFAULT);
  pinMode(PIN_PPS, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_PPS), on_pps, RISING);
  for (int i = 0; i < 8; i++) { analogRead(PIN_PANEL); analogRead(PIN_LIPO); delay(5); }
  Serial.println("# cloroplasto_E ready");
}

void loop() {
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
}