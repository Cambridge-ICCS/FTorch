title: FTorch Case Studies
author: Jack Atkinson
date: Last Updated: January 2026

FTorch has been deployed in a number of scientific projects.
This page collates the examples we are aware of.
If you have a case study you'd like to see listed here, please get in touch or open a pull request to add it!

If you make use of FTorch in your work please cite our publication:

<div style="margin-left: 2em;">
Atkinson et al., (2025). FTorch: a library for coupling PyTorch models to Fortran.<br>
<em>Journal of Open Source Software</em>, 10(107), 7602, <a href="https://doi.org/10.21105/joss.07602">https://doi.org/10.21105/joss.07602</a>
</div><br>

## Papers and Projects using FTorch

#### 2026

* Review paper of hybrid modelling approaches for cloud microphysics - [DOI: 10.1029/2025MS005341](https://doi.org/10.1029/2025MS005341)

#### 2025

* Testing winning parameterisation entries from the offline ClimSim Kaggle competition
  for online performance in `E3SM` -
  [DOI: 10.48550/arXiv.2511.20963](https://doi.org/10.48550/arXiv.2511.20963) (preprint)
* Development of a data-driven boundary layer momentum scheme in single-column `CAM`.
  Trained on LES schemes and outperforming existing parameterisations -
  [DOI: 10.48550/arXiv.2511.01766](https://doi.org/10.48550/arXiv.2511.01766) (preprint)
* By [UKAEA](https://www.gov.uk/government/organisations/uk-atomic-energy-authority)
  emulating turbulent transport models in fusion research - 
  [Code](https://github.com/ukaea/tglfnn-ukaea) - 
  [Conference paper](https://conferences.iaea.org/event/392/contributions/36059/)
* Emulation of cloud resolving models to reduce computational cost in `E3SM`.
  See Hu et al. (2025) - [DOI: 10.1029/2024MS004618](https://doi.org/10.1029/2024MS004618) (and [code](https://github.com/zyhu-hu/E3SM_nvlab/tree/ftorch/climsim_scripts/perlmutter_scripts))
* Review paper of hybrid modelling approaches -
  [DOI: 10.48550/arXiv.2510.03305](https://doi.org/10.48550/arXiv.2510.03305) (preprint)
* ClimSim Convection scheme in `ICON` giving a stable 20-year AMIP run -
  [DOI: 10.48550/arXiv.2510.08107](https://doi.org/10.48550/arXiv.2510.08107) (preprint)
* Bias correction of `CESM` through learning model biases compared to ERA5\
  [DOI: 10.1029/2024GL114106](https://doi.org/10.1029/2024GL114106)
* Implementation of nonlinear interactions in the `WaveWatch III` model -
  [DOI: 10.22541/essoar.174366388.80605654](https://doi.org/10.22541/essoar.174366388.80605654/v1) (preprint)
* In the [`GloSea6` Seasonal Forecasting Model](https://www.metoffice.gov.uk/research/climate/seasonal-to-decadal/gpc-outlooks/user-guide/global-seasonal-forecasting-system-glosea6) -
  Replacing a BiCGStab bottleneck in the code with a deep learning approach to speed up execution without compromising model accuracy.
  See Park and Chung (2025) - [DOI: 10.3390/atmos16010060](https://doi.org/10.3390/atmos16010060)

#### 2024

* [Convection parameterisations in ICON](https://github.com/EyringMLClimateGroup/heuer23_ml_convection_parameterization) -
  Implementing machine-learnt convection parameterisations in the `ICON` atmospheric model
  showing that best online performance occurs when causal relations are eliminated from the net.
  See Heuer et al (2024) - [DOI: 10.1029/2024MS004398](https://doi.org/10.1029/2024MS004398)
* [DataWave CAM-GW](https://github.com/DataWaveProject/CAM/) -
  Using FTorch to couple neural net parameterisations of gravity waves to the `CAM`
  atmospheric model.
* [MiMA Machine Learning](https://github.com/DataWaveProject/MiMA-machine-learning) -
  Implementing a neural net parameterisation of gravity waves in the `MiMA` atmospheric model.
  Demonstrates that nets trained near-identically offline can display greatly varied behaviours when coupled online.
  See Mansfield and Sheshadri (2024) - [DOI: 10.1029/2024MS004292](https://doi.org/10.1029/2024MS004292)
