# Photon Counter Correction and Protocol Metadata (Supplemental)

> **Status:** supplemental internal guide. This file is not part of the canonical documentation path.

This guide provides extra detail about two current product features:
1. photon-counter dead-time correction during load
2. protocol metadata capture during load

The canonical product path remains:
- `README.md`
- `docs/README.md`
- `docs/USER_MANUAL.md`
- `docs/mathematical_reference.md`

## Photon-counter dead-time correction

The current web application can apply a dead-time correction during file loading when the correction toggle is enabled.

### Correction model

$$
R_{true} = \frac{R_{measured}}{1 - R_{measured}\tau_D}
$$

where:
- $R_{measured}$ = measured count rate
- $\tau_D$ = detector dead time

### Typical use case

Use this correction when working with photon-counting detectors and when detector dead time can materially bias the measured count rate.

### Result metadata

Correction metadata is stored with the loaded result so downstream analysis can reference that provenance. This guide does not assume that every export surface reproduces the full correction metadata payload.

## Protocol metadata

Protocol information is captured on the load page and stored with the dataset metadata for internal traceability.

Typical fields include:
- measurement date / start time
- setup
- sample number
- particle
- concentration
- buffer and buffer concentration
- ND filter
- laser power
- stamp
- notes

## Scope note

This guide intentionally describes the current web workflow only. It does not serve as the canonical user manual and does not define public support or external issue-reporting channels.
