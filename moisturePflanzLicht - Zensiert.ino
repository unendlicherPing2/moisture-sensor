#include <ESPSupabase.h>
#include <WiFi.h>

Supabase db;

const char* ssid = "";
const char* password = "";
const String supabaseUrl = "";
const String supabaseKey = "";

//Time-to-Sleep (Every minute / 60s)
#define uS_TO_S_FACTOR 1000000ULL /* Conversion factor for micro seconds to seconds */
#define TIME_TO_SLEEP  30          /* Time ESP32 will go to sleep (in microseconds); multiplied by above conversion to achieve seconds*/

void setup() {
  Serial.begin(115200);

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    if (millis() % 500 == 0) Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  Serial.println(millis());

  db.begin(supabaseUrl, supabaseKey);
}

void loop() {
  analogReadResolution(11);
  int soilMoistureValue = analogRead(4); // Pin 4
  Serial.println(soilMoistureValue);
  //delay(1000);

  String jsonData = "{\"moisture_value\": " + String(soilMoistureValue) + "}";
  int httpCode = db.insert("moisture", jsonData, false);
  Serial.println(httpCode);
  db.urlQuery_reset();
  
  if (httpCode == 201) {
    Serial.println("Data inserted successfully");
  } else {
    Serial.println("Error inserting data");
  }
  
  Serial.println("Entering in Deep Sleep");
  esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * uS_TO_S_FACTOR /*/ 4*/);  // time set with variable above
  esp_deep_sleep_start();
}
