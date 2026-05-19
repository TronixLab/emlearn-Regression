#ifndef MINMAXSCALER_H
#define MINMAXSCALER_H

class MinMaxScaler {
public:
	static constexpr int n_features = 7;
	// Min and max values from scikit-learn fitting
	const float min[n_features] = { 3.8000f, 0.0800f, 0.0090f, 9.0000f, 0.98711f, 2.7200f, 8.0000f };
	const float max[n_features] = { 14.2000f, 1.1000f, 0.3460f, 440.0000f, 1.03898f, 3.8200f, 14.2000f };

	// Scale input array in-place
	void transform(float* X, float* X_scaled) const {
		for (int i = 0; i < n_features; ++i) {
			X_scaled[i] = (X[i] - min[i]) / (max[i] - min[i]);
		}
	}

	// Inverse scale input array in-place
	void inverse_transform(float* X_scaled, float* X) const {
		for (int i = 0; i < n_features; ++i) {
			X[i] = X_scaled[i] * (max[i] - min[i]) + min[i];
		}
	}
};

#endif  // MINMAXSCALER_H
