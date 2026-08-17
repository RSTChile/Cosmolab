/*
 * anima_tft_nrf24_terminal.ino
 * ------------------------------------------------------------
 * Micro-observatorio fisico ANIMA para Open-Smart UNO R3 Air
 * + Open Smart 3.5" ST7793 TFT.
 *
 * Pantallas:
 *   0 CABEZA  - duplicado 2D no espejado de la cabeza web
 *   1 RADIO   - terminal radio latente / paquetes discretos
 *   2 VIDA    - panel vital compacto
 *
 * Touch:
 *   toque corto: siguiente pantalla
 *   toque largo: volver a CABEZA
 *
 * Serial:
 *   Ttexto                 registra ultimo texto TX si radio esta activa
 *   N / P / H              pantalla siguiente / previa / cabeza
 *   H,theta,pitch,energia,OI,estado
 *   V,energia,hambre,saciedad,OI
 *   R,rx,tx,last_rx,last_tx,link
 *
 * Radio:
 *   Desactivada en esta variante estable de pantalla. La nRF24 por SPI
 *   hardware comparte pines con la TFT en Open-Smart y puede producir rayas.
 */

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

#define ENABLE_RF24 0
#if ENABLE_RF24
#include <RF24.h>
#endif

// El touch OPEN-SMART comparte pines de control de la TFT en este shield.
// Cuando TouchScreen::getPoint() se llama con la TFT activa, reconfigura A2/A3
// y puede corromper el bus paralelo: pantalla con rayas. Por eso el firmware
// estable deja la navegacion tactil desactivada y usa auto-ciclo + serial.
#define ENABLE_TOUCH_NAV 0
#if ENABLE_TOUCH_NAV
#include <TouchScreen.h>
#endif

const uint8_t PIN_RF_CE = A5;
const uint8_t PIN_RF_CSN = 5;
const uint8_t PIN_RF_IRQ = A4;

// Touch resistivo OPEN-SMART real validado por prueba de presión:
//   XP=8, YP=A3, XM=A2, YM=9
const int XP = 8, XM = A2, YP = A3, YM = 9;
const int MINPRESSURE = 10;
const int MAXPRESSURE = 1000;

MCUFRIEND_kbv tft;
#if ENABLE_RF24
RF24 radio(PIN_RF_CE, PIN_RF_CSN);
#endif
#if ENABLE_TOUCH_NAV
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);
#endif

const byte RF_ADDR_SELF[6] = "ANA01";
const byte RF_ADDR_PEER[6] = "ANE01";

const uint16_t C_BG = 0x080B;
const uint16_t C_PANEL = 0x18E3;
const uint16_t C_LINE = 0x2209;
const uint16_t C_TEXT = 0xFFFF;
const uint16_t C_MUTED = 0x9CF3;
const uint16_t C_GOLD = 0xDD65;
const uint16_t C_BLUE = 0x5D9F;
const uint16_t C_GREEN = 0x5EEC;
const uint16_t C_RED = 0xFBAA;
const uint16_t C_ORANGE = 0xFD20;
const uint16_t C_PURPLE = 0xA2FF;
const uint16_t C_HEAD = 0xDEFB;
const uint16_t C_IRIS = 0x03BF;

enum Screen { SCR_HEAD = 0, SCR_RADIO = 1, SCR_LIFE = 2, SCR_COUNT = 3 };
Screen screen = SCR_HEAD;
bool needsFullDraw = true;

bool rf_ok = false;
unsigned long rf_rx_count = 0;
unsigned long rf_tx_count = 0;
unsigned long last_status_ms = 0;
#if ENABLE_TOUCH_NAV
unsigned long last_touch_ms = 0;
#endif
unsigned long last_interaction_ms = 0;

char serial_cmd[64];
uint8_t serial_cmd_len = 0;
char last_rx[33] = "-";
char last_tx[33] = "-";
char last_tx_status[9] = "-";
char organism_state[18] = "listo";

float head_theta = 0.0;
float head_pitch = 0.0;
float vital_energy = 0.0;
float vital_oi = 0.0;
float vital_hunger = 0.0;
float vital_satiety = 0.0;
float radio_link = 0.0;

uint16_t display_id_raw = 0;
uint16_t display_id_used = 0x7793;

uint16_t chooseDisplayId(uint16_t raw) {
  return raw == 0x7793 ? raw : 0x7793;
}

void setScreen(Screen s) {
  screen = s;
  needsFullDraw = true;
  last_interaction_ms = millis();
}

void nextScreen() {
  setScreen((Screen)((screen + 1) % SCR_COUNT));
}

void prevScreen() {
  setScreen((Screen)((screen + SCR_COUNT - 1) % SCR_COUNT));
}

#if ENABLE_TOUCH_NAV
void restoreTouchPins() {
  pinMode(YP, OUTPUT);
  pinMode(XM, OUTPUT);
  pinMode(XP, OUTPUT);
  pinMode(YM, OUTPUT);
  digitalWrite(YP, HIGH);
  digitalWrite(XM, HIGH);
  digitalWrite(XP, HIGH);
  digitalWrite(YM, HIGH);
}

bool touchPressed(bool *longPress) {
  TSPoint p = ts.getPoint();
  restoreTouchPins();
  bool down = (p.z > MINPRESSURE && p.z < MAXPRESSURE);
  if (!down) return false;
  unsigned long now = millis();
  if (now - last_touch_ms < 450) return false;
  last_touch_ms = now;
  Serial.print(F("TOUCHRAW,"));
  Serial.print(p.x);
  Serial.print(',');
  Serial.print(p.y);
  Serial.print(',');
  Serial.println(p.z);
  *longPress = p.z > 650;
  return true;
}
#endif

void header(const __FlashStringHelper *title, uint16_t accent) {
  tft.fillRect(0, 0, tft.width(), 34, C_PANEL);
  tft.drawFastHLine(0, 34, tft.width(), C_LINE);
  tft.setTextSize(2);
  tft.setTextColor(C_TEXT, C_PANEL);
  tft.setCursor(10, 9);
  tft.print(title);
  tft.fillRect(tft.width() - 66, 8, 44, 16, accent);
  tft.setTextSize(1);
  tft.setTextColor(C_BG, accent);
  tft.setCursor(tft.width() - 58, 13);
  tft.print(screen + 1);
  tft.print(F("/3"));
  tft.setTextColor(C_MUTED, C_PANEL);
  tft.setCursor(tft.width() - 116, 25);
  tft.print(F("OBS v2"));
}

void row(int y, const __FlashStringHelper *k, const char *v, uint16_t col = C_TEXT) {
  tft.fillRect(10, y, tft.width() - 20, 16, C_BG);
  tft.setTextSize(1);
  tft.setTextColor(C_MUTED, C_BG);
  tft.setCursor(12, y + 3);
  tft.print(k);
  tft.setTextColor(col, C_BG);
  int16_t x = tft.width() - (strlen(v) * 6) - 12;
  if (x < 120) x = 120;
  tft.setCursor(x, y + 3);
  tft.print(v);
}

void rowNum(int y, const __FlashStringHelper *k, float value, uint8_t dec, const char *suffix = "", uint16_t col = C_TEXT) {
  char buf[22];
  dtostrf(value, 0, dec, buf);
  strncat(buf, suffix, sizeof(buf) - strlen(buf) - 1);
  row(y, k, buf, col);
}

void gauge(int x, int y, int w, int h, float v, uint16_t col) {
  if (v < 0) v = 0;
  if (v > 1) v = 1;
  tft.drawRect(x, y, w, h, C_LINE);
  tft.fillRect(x + 1, y + 1, w - 2, h - 2, 0x1020);
  tft.fillRect(x + 1, y + 1, (int)((w - 2) * v), h - 2, col);
}

void banner(int y, const char *txt, uint16_t bg, uint16_t fg) {
  tft.fillRoundRect(10, y, tft.width() - 20, 30, 5, bg);
  tft.drawRoundRect(10, y, tft.width() - 20, 30, 5, C_LINE);
  tft.setTextSize(2);
  tft.setTextColor(fg, bg);
  tft.setCursor(18, y + 8);
  tft.print(txt);
}

void drawHeadIcon() {
  const int cx = tft.width() / 2;
  const int cy = 130;
  float yaw = head_theta;
  if (yaw < -45) yaw = -45;
  if (yaw > 45) yaw = 45;
  float pitch = head_pitch;
  if (pitch < -25) pitch = -25;
  if (pitch > 25) pitch = 25;
  int dx = (int)(yaw * 0.42);
  int dy = (int)(-pitch * 0.30);

  // Reference-like floating base / neck.
  tft.fillRoundRect(cx - 4, cy + 66, 8, 34, 4, 0x05E0);
  tft.fillRoundRect(cx - 2, cy + 68, 4, 30, 2, C_GREEN);

  // Side lobes first, behind the head.
  tft.fillCircle(cx - 70, cy - 4, 23, 0xBDF7);
  tft.fillCircle(cx + 70, cy - 4, 23, 0xBDF7);
  tft.fillCircle(cx - 65, cy - 4, 20, 0xE71C);
  tft.fillCircle(cx + 65, cy - 4, 20, 0xE71C);
  tft.drawCircle(cx - 70, cy - 4, 23, C_LINE);
  tft.drawCircle(cx + 70, cy - 4, 23, C_LINE);

  // White sphere with simple faux shading.
  tft.fillCircle(cx, cy, 68, 0xCE79);
  tft.fillCircle(cx - 6, cy - 8, 66, 0xE71C);
  tft.fillCircle(cx - 18, cy - 20, 56, 0xFFFF);
  tft.fillCircle(cx + 28, cy + 36, 22, 0xC638);
  tft.fillCircle(cx + 22, cy - 26, 14, 0xFFFF);
  tft.drawCircle(cx, cy, 68, C_LINE);

  int eyeY = cy - 6 + dy;
  int lx = cx - 24 + dx;
  int rx = cx + 24 + dx;
  int pupilShift = dx / 7;

  // Soft eye sockets.
  tft.fillCircle(lx - 2, eyeY + 2, 17, 0xBDF7);
  tft.fillCircle(rx - 2, eyeY + 2, 17, 0xBDF7);
  tft.fillCircle(lx, eyeY, 15, 0xE71C);
  tft.fillCircle(rx, eyeY, 15, 0xE71C);
  tft.fillCircle(lx, eyeY, 11, C_TEXT);
  tft.fillCircle(rx, eyeY, 11, C_TEXT);
  tft.fillCircle(lx + pupilShift, eyeY, 7, C_IRIS);
  tft.fillCircle(rx + pupilShift, eyeY, 7, C_IRIS);
  tft.fillCircle(lx + pupilShift, eyeY, 4, 0x0012);
  tft.fillCircle(rx + pupilShift, eyeY, 4, 0x0012);
  tft.fillCircle(lx + pupilShift - 3, eyeY - 4, 2, C_TEXT);
  tft.fillCircle(rx + pupilShift - 3, eyeY - 4, 2, C_TEXT);

  // Small sad mouth, like the web head.
  int mouthY = cy + 36 + dy / 2;
  uint16_t mouthCol = 0x4208;
  tft.drawLine(cx - 18 + dx / 6, mouthY + 5, cx - 8 + dx / 8, mouthY, mouthCol);
  tft.drawLine(cx - 8 + dx / 8, mouthY, cx + 8 + dx / 8, mouthY, mouthCol);
  tft.drawLine(cx + 8 + dx / 8, mouthY, cx + 18 + dx / 6, mouthY + 5, mouthCol);
}

void drawHeadScreen() {
  header(F("OBSERVATORIO CABEZA"), C_BLUE);
  banner(42, organism_state, 0x1224, C_GREEN);
  drawHeadIcon();
  rowNum(202, F("yaw"), head_theta, 1, " deg", C_BLUE);
  rowNum(218, F("pitch"), head_pitch, 1, " deg", C_PURPLE);
  gauge(10, 236, tft.width() - 20, 8, vital_oi, C_GREEN);
}

void drawRadioScreen() {
  header(F("RADIO DIGITAL"), C_BLUE);
#if ENABLE_RF24
  bool chip = rf_ok && radio.isChipConnected();
  radio_link = chip ? 1.0 : 0.0;
#else
  bool chip = false;
  radio_link = 0.0;
#endif
  banner(42, chip ? "enlace local OK" : "radio no responde", chip ? 0x1224 : 0x3008, chip ? C_GREEN : C_RED);
  row(84, F("chip SPI"), chip ? "si" : "no", chip ? C_GREEN : C_RED);
  rowNum(102, F("recibidos"), rf_rx_count, 0, "", C_BLUE);
  rowNum(120, F("enviados"), rf_tx_count, 0, "", C_ORANGE);
  row(146, F("ultimo RX"), last_rx, C_TEXT);
  row(164, F("ultimo TX"), last_tx, C_TEXT);
  row(182, F("estado TX"), last_tx_status, strcmp(last_tx_status, "ok") == 0 ? C_GREEN : C_ORANGE);
  gauge(10, 212, tft.width() - 20, 10, radio_link, C_BLUE);
  tft.setTextSize(1);
  tft.setTextColor(C_MUTED, C_BG);
  tft.setCursor(12, 230);
  tft.print(F("touch: siguiente  |  serial: Ttexto"));
}

void drawLifeScreen() {
  header(F("VIDA / METABOLISMO"), C_GOLD);
  banner(42, organism_state, 0x2418, C_GOLD);
  rowNum(86, F("energia"), vital_energy, 3, "", C_GOLD);
  gauge(10, 104, tft.width() - 20, 9, vital_energy, C_GOLD);
  rowNum(124, F("hambre"), vital_hunger, 3, "", C_ORANGE);
  gauge(10, 142, tft.width() - 20, 9, vital_hunger, C_ORANGE);
  rowNum(162, F("saciedad"), vital_satiety, 3, "", C_GREEN);
  gauge(10, 180, tft.width() - 20, 9, vital_satiety, C_GREEN);
  rowNum(202, F("OI"), vital_oi, 3, "", C_GREEN);
  gauge(10, 220, tft.width() - 20, 10, vital_oi, C_GREEN);
}

void drawScreen() {
  tft.fillScreen(C_BG);
  if (screen == SCR_HEAD) drawHeadScreen();
  else if (screen == SCR_RADIO) drawRadioScreen();
  else drawLifeScreen();
  needsFullDraw = false;
}

void refreshScreen() {
  if (needsFullDraw) {
    drawScreen();
    return;
  }
  if (screen == SCR_RADIO) drawRadioScreen();
}

void setupDisplay() {
  display_id_raw = tft.readID();
  display_id_used = chooseDisplayId(display_id_raw);
  tft.begin(display_id_used);
  tft.setRotation(1);
}

void setupRadio() {
#if ENABLE_RF24
  pinMode(PIN_RF_IRQ, INPUT_PULLUP);
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);
  rf_ok = radio.begin();
  if (rf_ok) {
    radio.setPALevel(RF24_PA_LOW);
    radio.setDataRate(RF24_1MBPS);
    radio.setChannel(76);
    radio.setPayloadSize(32);
    radio.openWritingPipe(RF_ADDR_PEER);
    radio.openReadingPipe(1, RF_ADDR_SELF);
    radio.startListening();
  }
  Serial.print(F("# RF24: "));
  Serial.print(rf_ok ? F("OK") : F("FAIL"));
  Serial.print(F(" connected="));
  Serial.println(radio.isChipConnected() ? 1 : 0);
#else
  rf_ok = false;
  Serial.println(F("# RF24: disabled for TFT-only mode"));
#endif
}

void sendRadioText(const char *text) {
  char payload[32];
  memset(payload, 0, sizeof(payload));
  strncpy(payload, text, sizeof(payload) - 1);
  strncpy(last_tx, payload[0] ? payload : "-", sizeof(last_tx) - 1);
  last_tx[sizeof(last_tx) - 1] = 0;

  bool ok = false;
#if ENABLE_RF24
  if (rf_ok) {
    radio.stopListening();
    ok = radio.write(payload, sizeof(payload));
    radio.startListening();
  }
#endif
  rf_tx_count++;
  strncpy(last_tx_status, ok ? "ok" : "fail", sizeof(last_tx_status) - 1);
  last_tx_status[sizeof(last_tx_status) - 1] = 0;
  Serial.print(F("RFTX,"));
  Serial.println(ok ? F("ok") : F("fail"));
  needsFullDraw = true;
}

char *nextField(char **s) {
  char *p = *s;
  if (!p) return NULL;
  char *comma = strchr(p, ',');
  if (comma) {
    *comma = 0;
    *s = comma + 1;
  } else {
    *s = NULL;
  }
  return p;
}

void parseStateCommand(char *cmd) {
  char tag = cmd[0];
  char *p = cmd + 2;
  if (tag == 'H') {
    char *f;
    if ((f = nextField(&p))) head_theta = atof(f);
    if ((f = nextField(&p))) head_pitch = atof(f);
    if ((f = nextField(&p))) vital_energy = atof(f);
    if ((f = nextField(&p))) vital_oi = atof(f);
    if ((f = nextField(&p))) {
      strncpy(organism_state, f, sizeof(organism_state) - 1);
      organism_state[sizeof(organism_state) - 1] = 0;
    }
    needsFullDraw = true;
  } else if (tag == 'V') {
    char *f;
    if ((f = nextField(&p))) vital_energy = atof(f);
    if ((f = nextField(&p))) vital_hunger = atof(f);
    if ((f = nextField(&p))) vital_satiety = atof(f);
    if ((f = nextField(&p))) vital_oi = atof(f);
    needsFullDraw = true;
  } else if (tag == 'R') {
    char *f;
    if ((f = nextField(&p))) rf_rx_count = atol(f);
    if ((f = nextField(&p))) rf_tx_count = atol(f);
    if ((f = nextField(&p))) strncpy(last_rx, f, sizeof(last_rx) - 1);
    if ((f = nextField(&p))) strncpy(last_tx, f, sizeof(last_tx) - 1);
    if ((f = nextField(&p))) radio_link = atof(f);
    last_rx[sizeof(last_rx) - 1] = 0;
    last_tx[sizeof(last_tx) - 1] = 0;
    needsFullDraw = true;
  }
}

void handleCommand() {
  serial_cmd[serial_cmd_len] = 0;
  if (serial_cmd_len == 0) return;
  if (serial_cmd[0] == 'T') {
    sendRadioText(&serial_cmd[1]);
  } else if (serial_cmd[0] == 'N') {
    nextScreen();
    Serial.println(F("CMD,N"));
  } else if (serial_cmd[0] == 'P') {
    prevScreen();
    Serial.println(F("CMD,P"));
  } else if (serial_cmd[0] == 'C') {
    setScreen(SCR_HEAD);
    Serial.println(F("CMD,C"));
  } else if ((serial_cmd[0] == 'H' || serial_cmd[0] == 'V' || serial_cmd[0] == 'R') && serial_cmd[1] == ',') {
    parseStateCommand(serial_cmd);
    Serial.print(F("CMD,"));
    Serial.println(serial_cmd[0]);
  }
}

void handleSerialChar(char c) {
  if (c == '\n' || c == '\r') {
    handleCommand();
    serial_cmd_len = 0;
  } else if (serial_cmd_len < sizeof(serial_cmd) - 1) {
    serial_cmd[serial_cmd_len++] = c;
  } else {
    serial_cmd_len = 0;
  }
}

void readRadio() {
#if ENABLE_RF24
  if (!rf_ok) return;
  while (radio.available()) {
    char buf[33];
    memset(buf, 0, sizeof(buf));
    radio.read(buf, 32);
    rf_rx_count++;
    strncpy(last_rx, buf[0] ? buf : "-", sizeof(last_rx) - 1);
    last_rx[sizeof(last_rx) - 1] = 0;
    Serial.print(F("RFRX,"));
    Serial.println(last_rx);
    needsFullDraw = true;
  }
#endif
}

void reportStatus() {
  unsigned long now = millis();
  if (now - last_status_ms < 1000UL) return;
  last_status_ms = now;
  Serial.print(F("RF24,"));
  Serial.print(rf_ok ? 1 : 0);
  Serial.print(',');
#if ENABLE_RF24
  Serial.print(radio.isChipConnected() ? 1 : 0);
#else
  Serial.print(0);
#endif
  Serial.print(',');
  Serial.print(rf_rx_count);
  Serial.print(',');
  Serial.println(rf_tx_count);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  setupDisplay();
  setupRadio();
  Serial.print(F("# TFT: raw=0x"));
  Serial.print(display_id_raw, HEX);
  Serial.print(F(" used=0x"));
  Serial.println(display_id_used, HEX);
  Serial.print(F("# IRQ A4="));
  Serial.println(digitalRead(PIN_RF_IRQ) ? 1 : 0);
  Serial.println(F("# anima_tft_micro_observatorio ready"));
  last_interaction_ms = millis();
  drawScreen();
}

void loop() {
  while (Serial.available()) handleSerialChar((char)Serial.read());
#if ENABLE_TOUCH_NAV
  bool longPress = false;
  if (touchPressed(&longPress)) {
    if (longPress) {
      setScreen(SCR_HEAD);
      Serial.println(F("TOUCH,C"));
    } else {
      nextScreen();
      Serial.println(F("TOUCH,N"));
    }
  }
#endif
  if (millis() - last_interaction_ms > 7000UL) {
    nextScreen();
  }
  readRadio();
  reportStatus();
  refreshScreen();
}
