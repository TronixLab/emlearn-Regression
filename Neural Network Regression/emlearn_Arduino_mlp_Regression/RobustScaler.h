#ifndef ROBUSTSCALER_H
#define ROBUSTSCALER_H

class RobustScaler {
public:
	static constexpr int n_features = 7;
	// Median and IQR values from scikit-learn fitting
	const float median[n_features] = {0.52000f, 0.26000f, 0.079000f, 38.0000f, 0.99675f, 0.62000f, 10.2000f};
	const float iqr[n_features] = {0.2500f, 0.3300f, 0.02000f, 40.0000f, 0.002235f, 0.1800f, 1.6000f};

	// Scale input array in-place
	void transform(float* X, float* X_scaled) const {
		for (int i = 0; i < n_features; ++i) {
			X_scaled[i] = (X[i] - median[i]) / iqr[i];
		}
	}

	// Inverse scale input array in-place
	void inverse_transform(float* X_scaled, float* X) const {
		for (int i = 0; i < n_features; ++i) {
			X[i] = X_scaled[i] * iqr[i] + median[i];
		}
	}
};

#endif // ROBUSTSCALER_H
