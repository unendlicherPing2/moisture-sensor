#define Second int

const char MOISTURE_SENSOR = 36;
const int uS_TO_S_FACTOR = 1000000ULL;
const Second MESSURE_EVERY = 1;
const int MAX_TELEMETRY = 1;

typedef struct {
  int time;
  int data;
} Telemetry;

RTC_DATA_ATTR Telemetry telemetryData[MAX_TELEMETRY] = {};
RTC_DATA_ATTR int telemetry_count = 0;


constexpr int get_average(int values[5]) {
  return (values[0] + values[1] + values[2] + values[3] + values[4]) / 5;
}

void setup() {
  Serial.begin(9600);
  analogReadResolution(8);
  esp_sleep_enable_timer_wakeup(MESSURE_EVERY * uS_TO_S_FACTOR);

  int messurments[5] = {};
  for (int i = 0; i < 5; i++) {
    messurments[i] = analogRead(MOISTURE_SENSOR);
    delay(100);
  }

  int average = get_average(messurments);

  telemetryData[telemetry_count].time = millis();
  telemetryData[telemetry_count].data = average;
  telemetry_count++;

  if (telemetry_count >= MAX_TELEMETRY) {
    Serial.println(F("Telemetry: "));

    for (int i = 0; i < telemetry_count; i++) print_telemetry(i, telemetryData[i]);

    telemetry_count = 0;
  }

  Serial.flush();
  esp_deep_sleep_start();
}

void print_telemetry(int index, Telemetry telemetry) {
  Serial.printf("%d. %d: %d\n", index + 1, telemetry.time, telemetry.data);
}

void loop() {}
