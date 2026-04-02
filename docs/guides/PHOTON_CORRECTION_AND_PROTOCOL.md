# Photon Counter Correction and Protocol Metadata

## Overview

The Peak Analysis Tool now includes two important features for enhanced data quality and experiment documentation:

1. **Photon Counter Dead-Time Correction**: Compensates for detector non-linearity
2. **Protocol Information**: Captures experiment metadata for reproducibility

These features are available in both the web UI (`/load` page) and the legacy Tkinter interface.

---

## Photon Counter Dead-Time Correction

### What is Dead-Time Correction?

Photon counting detectors have a characteristic "dead time" (T_D) during which they cannot register new photons after detecting one. This causes the measured count rate to underestimate the true photon rate, especially at high count rates.

### The Correction Formula

The correction compensates for this non-linearity using:

```
corrected_counts = measured_counts × 1 / (1 - R_measured × T_D)
```

Where:
- **R_measured** = measured count rate (counts per second) = counts / dwell_time
- **T_D** = detector dead time (seconds)

### When to Use It

Apply dead-time correction when:
- Working with photon counting detectors (PMTs, APDs, SPADs)
- Count rates approach or exceed 1 MHz
- Accurate quantification is critical
- Comparing measurements at different count rates

### How to Use (Web UI)

1. Navigate to the **Load Data** page
2. Expand the **Photon Counter Correction** card
3. Toggle **Apply dead-time correction** to ON
4. Set the **Dead time** value (default: 43 ns for typical APD)
5. Load your data files

The correction is applied automatically during data loading.

### How to Use (Legacy Tkinter UI)

1. Open the **Data Loading** tab
2. Check **Apply dead-time correction on load**
3. Enter the **Dead time (ns)** value
4. Load your data files

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Dead time (T_D) | 43 ns | 1-1000 ns | Detector dead time period |

**Common Values:**
- APD (Avalanche Photodiode): 30-50 ns
- PMT (Photomultiplier Tube): 10-100 ns
- SPAD (Single Photon Avalanche Diode): 20-80 ns

### Understanding the Results

After correction, the metadata includes:

```json
{
  "corrections": {
    "applied": true,
    "dead_time_ns": 43.0,
    "max_correction_factor": 1.234,
    "mean_correction_factor": 1.045,
    "saturated_points": 0,
    "max_count_rate_hz": 5000000
  }
}
```

**Key Metrics:**
- **mean_correction_factor**: Average correction applied (e.g., 1.05 = 5% increase)
- **max_correction_factor**: Maximum correction (capped at 10× for numerical stability)
- **saturated_points**: Number of points near detector saturation
- **max_count_rate_hz**: Peak observed count rate

### Saturation Warning

When the correction factor exceeds 10× (denominator < 0.1), points are clamped to avoid numerical instability. This corresponds to ~90% of theoretical saturation:

```
Saturation occurs when: R_measured × T_D ≈ 1
```

For T_D = 43 ns, saturation begins around 23 MHz count rate.

### Example Calculation

**Scenario:** 
- Measured counts: 1000
- Dwell time: 0.1 ms (1e-4 s)
- Dead time: 43 ns (43e-9 s)

**Calculation:**
1. Count rate: R = 1000 / 1e-4 = 10 MHz
2. Denominator: 1 - (10e6 × 43e-9) = 1 - 0.43 = 0.57
3. Correction factor: 1 / 0.57 = 1.75
4. Corrected counts: 1000 × 1.75 = 1750

This represents a 75% correction due to dead-time losses.

---

## Protocol Information

### Purpose

Protocol metadata captures experimental conditions for:
- **Reproducibility**: Document exact experimental setup
- **Traceability**: Link data to specific samples and conditions
- **Analysis**: Group and compare measurements by protocol parameters
- **Publication**: Include complete methods in supplementary materials

### Available Fields

| Field | Description | Example |
|-------|-------------|---------|
| Measurement Date | Date of data acquisition | 2025-01-15 |
| Start Time | Experiment start time | 14:30:00 |
| Setup | Instrument configuration | "Prototype, Old Ladom" |
| Sample Number | Sample identifier | "S-2025-001" |
| Particle | Particle or analyte type | "DNA oligomer" |
| Concentration | Particle concentration | "10 nM" |
| Buffer | Buffer solution | "PBS" |
| Buffer Concentration | Buffer molarity | "10 mM, pH 7.4" |
| ND Filter | Neutral density filter | "ND 2.0" |
| Laser Power | Excitation power | "5 mW" |
| Stamp | Lithographic stamp ID | "triple-block" |
| Notes | Additional observations | "Sample showed aggregation" |

### How to Use (Web UI)

1. Navigate to the **Load Data** page
2. Scroll to the **Protocol Information** card
3. Fill in relevant fields (all optional)
4. Load your data files

Protocol information is saved with the data and can be exported with results.

### How to Use (Legacy Tkinter UI)

1. Open the **Data Loading** tab
2. Scroll to the **Protocol Information** section
3. Fill in the form fields
4. Load your data files

### Best Practices

**Essential Fields:**
- Measurement Date
- Sample Number
- Particle/Sample type
- Concentration

**Recommended Fields:**
- Setup (for multi-instrument labs)
- Buffer conditions
- Laser power
- Notes (for anomalies)

**Optional Fields:**
- ND Filter (if variable)
- Stamp (for nanofabrication experiments)
- Start Time (for time-series experiments)

### Accessing Protocol Data

**In Python (Backend):**
```python
# Protocol is stored in result metadata
result = store.get_result(result_id)
protocol = result['meta'].get('protocol')
if protocol:
    print(f"Sample: {protocol['sample_number']}")
    print(f"Concentration: {protocol['concentration']}")
```

**In TypeScript (Frontend):**
```typescript
// Protocol is in the meta object
const { meta } = useDataStore()
if (meta?.protocol) {
  console.log('Sample:', meta.protocol.sample_number)
  console.log('Particle:', meta.protocol.particle)
}
```

**In CSV Export:**
Protocol fields are included as header rows in exported CSV files.

---

## Integration with Analysis Pipeline

### Data Flow

```
1. User fills protocol form + enables correction
2. Files uploaded to backend
3. load_data_from_paths() called with options
4. If correction enabled:
   - apply_dead_time_correction() modifies amplitude
   - Correction metadata stored
5. Protocol dict stored in result metadata
6. Both available in all downstream analysis
```

### Accessing in Analysis

**Peak Detection:**
```python
# Correction info available in result meta
result = store.get_result(result_id)
corrections = result['meta'].get('corrections')
if corrections and corrections.get('applied'):
    print(f"Data was corrected with factor: {corrections['mean_correction_factor']}")
```

**Export:**
```python
# Protocol included in CSV exports
df = export_peaks_to_csv_data(...)
# Add protocol as header comments
with open('output.csv', 'w') as f:
    if protocol:
        f.write(f"# Sample: {protocol['sample_number']}\n")
        f.write(f"# Date: {protocol['measurement_date']}\n")
    df.to_csv(f, index=False)
```

---

## Testing and Validation

### Unit Tests

Photon correction is tested in `tests/unit/test_photon_correction.py`:

```bash
# Run correction tests
pytest tests/unit/test_photon_correction.py -v
```

**Test Coverage:**
- Basic correction formula
- Zero signal handling
- Low and high count rates
- Saturation clamping
- Different dead times
- Edge cases

### Manual Testing

**Test Correction:**
1. Load sample data with known count rates
2. Enable correction with T_D = 43 ns
3. Verify correction factors are reasonable (1.0-2.0 for typical data)
4. Check for saturation warnings at high rates

**Test Protocol:**
1. Fill all protocol fields
2. Load data
3. Verify protocol appears in metadata
4. Export CSV and check header includes protocol

### Validation Checklist

- [ ] Correction increases signal amplitude
- [ ] Correction factor > 1.0 for all non-zero points
- [ ] No saturation warnings for normal data
- [ ] Protocol fields persist across sessions (Zustand)
- [ ] Protocol included in exported files
- [ ] Both features work in web and legacy UI

---

## Troubleshooting

### Correction Issues

**Problem:** Correction factor too high (>3×)
- **Cause:** Count rate approaching saturation
- **Solution:** Reduce laser power or use ND filter

**Problem:** Many saturated points warning
- **Cause:** Count rate exceeds ~90% of saturation limit
- **Solution:** Dilute sample or reduce excitation

**Problem:** Correction has no effect
- **Cause:** Count rates too low (correction ≈ 1.0)
- **Solution:** This is normal; correction only significant at MHz rates

### Protocol Issues

**Problem:** Protocol fields not saving
- **Cause:** Browser storage disabled
- **Solution:** Enable localStorage in browser settings

**Problem:** Protocol not in export
- **Cause:** Old backend version
- **Solution:** Update to latest version with protocol support

---

## References

### Dead-Time Correction

1. Müller, J. D. (2004). "Cumulant analysis in fluorescence fluctuation spectroscopy." *Biophysical Journal*, 86(5), 3981-3992.

2. Laurence, T. A., et al. (2006). "Photon arrival-time interval distribution (PAID): a novel tool for analyzing molecular interactions." *Journal of Physical Chemistry B*, 110(19), 9764-9770.

3. Wahl, M., et al. (2003). "Dead-time optimized time-correlated photon counting instrument with synchronized, independent timing channels." *Review of Scientific Instruments*, 74(3), 1948-1955.

### Best Practices

4. Enderlein, J., & Gregor, I. (2005). "Using fluorescence lifetime for discriminating detector afterpulsing in fluorescence-correlation spectroscopy." *Review of Scientific Instruments*, 76(3), 033102.

---

## Version History

- **v1.3.0** (2025-01-15): Initial implementation
  - Added photon counter dead-time correction
  - Added protocol metadata capture
  - Integrated with web UI and legacy UI
  - Full test coverage

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/yourusername/peak-analysis-tool/issues
- Email: support@yourlab.edu
- Documentation: See main user manual for general usage


