#include <Arduino_LSM9DS1.h>

void setup() {
  Serial.begin(9600);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("Gyroscope Ready");
}

void loop() {
  float x, y, z;

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(x, y, z);

    Serial.print("X = ");
    Serial.print(x);

    Serial.print("  Y = ");
    Serial.print(y);

    Serial.print("  Z = ");
    Serial.println(z);
  }
}