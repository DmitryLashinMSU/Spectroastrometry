# Spectroastrometry
The repository contains a <strong><a href="./Spectroastrometry.py">Python module</a></strong> with a set of functions for spectroastrometric data processing, as well as a <strong><a href="./Example/Example.ipynb">usage example</a></strong>. The method is described in the paper [Whelan & Garcia 2008](https://ui.adsabs.harvard.edu/abs/2008LNP...742..123W/abstract).

## Quick Start
Place the `Spectroastrometry.py` file in your Python project directory and import it as a standard library: `Spectroastrometry`.

## Documentation
Below is a list of the module's functions, their descriptions, and parameters.

---

### `G_fit`
Approximation with a "positive" Gaussian function for the intensity profile.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `coord`   | array of coordinates (pixels) |
| `I`       | array of intensities (ADU) |

**Returns:**
- `P` — array of model parameters:  
  `P[0]` — integrated intensity,  
  `P[1]` — standard deviation,  
  `P[2]` — center position,  
  `P[3]` — background.

**Note:**  
`G_fit` uses the error function `erf` to model a discrete Gaussian profile.

---

### `neg_G_fit`
Approximation with a "negative" Gaussian function (for absorption lines).

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `coord`   | array of coordinates (pixels) |
| `I`       | array of intensities (ADU) |

**Returns:**
- `P` — array of model parameters (similar to `G_fit`, but for absorption).

---

### `Center_search`
Calculation of the observed shift and FWHM along the spectral slice for all spectral points along the dispersion direction.

**Arguments:**
| Parameter  | Description |
|------------|-------------|
| `CRVAL1`   | starting wavelength value of the spectrum (λ) |
| `CDELT1`   | dispersion (Δλ/pixel) |
| `x_start`  | starting X pixel |
| `x_end`    | ending X pixel |
| `area`     | half-width of the selected analysis region along Y |
| `Y_est`    | estimated spectrum center along Y |
| `image`    | 2D image array |

**Returns:**
- `LAMBDA`   — array of wavelengths
- `SPEC`     — spectrum
- `CENTER`   — center coordinates along Y for each X
- `FWHM`     — full width at half maximum for each X
- `ERRORBAR` — uncertainty in the center determination

**Note:**  
The uncertainty is calculated according to [Condon, 1977](https://ui.adsabs.harvard.edu/abs/1997PASP..109..166C/abstract) using the formula:

$$
\delta = 0.6 \frac{FWHM}{SNR},
$$

where $FWHM$ is the full width at half maximum, $SNR$ is the signal-to-noise ratio.

---

### `LCS_auto`
Estimate of the true shift based on a spectral line (automatic line center search). This is done by computing the brightness profiles of the spectrum in the region of the selected line's maximum, as well as with some offset from it (where only the continuum is present). Subtracting the continuum yields the pure line profile, whose center corresponds to the true position of the source emitting in this line.

**Arguments:**
| Parameter   | Description |
|-------------|-------------|
| `CRVAL1`    | starting λ value |
| `CDELT1`    | dispersion (Δλ/pixel) |
| `image`     | 2D image |
| `LAMBDA`    | array of wavelengths (from `Center_search`) |
| `SPEC`      | spectrum (from `Center_search`) |
| `x_start`   | line start (pixel) |
| `x_end`     | line end (pixel) |
| `x_cont`    | X-coordinate of the point selected for continuum determination |
| `Y_est`     | estimated center along Y |
| `area`      | half-width of the region along Y |
| `Line_type` | line type: `'emission'` or `'absorption'` |

**Returns:**
- `P4` — parameters of the line model after continuum subtraction
- Position of the line center in pixels

---

### `LCS_target`
Estimate of the true shift based on a line with a manually specified center.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `image`   | 2D image |
| `x_line`  | pixel of the line center |
| `x_cont`  | pixel for continuum |
| `Y_est`   | estimated center along Y |
| `area`    | half-width of the region along Y |

**Returns:**
- `P4` — parameters of the line model
- `P3` — parameters of the background (continuum)

---

### `Median_images`
Median processing of a set of images.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `images`  | list of 2D arrays of the same size |

**Returns:**
- `median_image` — median image

---

### `Average_images`
Averaging of a set of images.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `images`  | list of 2D arrays of the same size |

**Returns:**
- `average_image` — averaged image

---

### `Detrend`
Trend removal using a rolling mean method.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `window`  | rolling mean window width |
| `CENTER`  | array of center values |

**Returns:**
- `CENTER_rolling` — centers after trend removal
- `rolling_mean`   — rolling mean

---

### `remove_polynomial_trend`
Removal of an n-th degree polynomial trend.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `data`    | 2D data array (e.g., `CENTER`) |
| `n`       | polynomial degree |

**Returns:**
- `detrended_data` — detrended data

---

### `show_3D`
Plots a 3D graph of a selected region of the image.

**Arguments:**
| Parameter | Description |
|-----------|-------------|
| `image`   | 2D image |
| `x_start` | start X |
| `x_end`   | end X |
| `y_start` | start Y |
| `y_end`   | end Y |

---

## Dependencies
- `numpy`
- `matplotlib.pyplot`
- `scipy.optimize.least_squares`
- `scipy.special.erf`

---

>Contact me: [lashinda@my.msu.ru](mailto:lashinda@my.msu.ru)
