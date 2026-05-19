#include "ExtraTreesRegressor.h"
#include "MinMaxScaler.h"

// Create scaler instance
MinMaxScaler scaler;

// Array to hold scaled features
float features_scaled[MinMaxScaler::n_features];

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    float features[7];
    int featureIndex = 0;
    int lastIndex = 0;
    line.trim();

    while (featureIndex < 7) {
      int commaIndex = line.indexOf(',', lastIndex);
      String value;
      if (commaIndex == -1) {
        value = line.substring(lastIndex);
      } else {
        value = line.substring(lastIndex, commaIndex);
      }
      features[featureIndex] = value.toFloat();
      featureIndex++;
      if (commaIndex == -1) break;
      lastIndex = commaIndex + 1;
    }

    // Scale features
    scaler.transform(features, features_scaled);

    // Make prediction
    float y_pred = ExtraTreesRegressor_predict(features_scaled, sizeof(features_scaled) / sizeof(features_scaled[0]));

    // Print prediction
    Serial.print("Prediction: ");
    Serial.println(y_pred);
  }
  delay(100);
}