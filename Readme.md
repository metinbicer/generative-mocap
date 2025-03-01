# Overview

- We trained generative adversarial networks (GANs) with conditions for synthetic gait data generation (marker, joint angles and ground reaction forces/moments) and used the generated data to augment the existing experimental datasets.

- ***Check out the [web app](https://thisgaitdoesnotexist.streamlit.app/) where we deploy our generative models to generate, animate and download synthetic data.***

- Please refer to our approach explained in the following papers:
```bibtex
@article{Bicer2022Generative,
  title={Generative Deep Learning Applied to Biomechanics: A New Augmentation Technique for Motion Capture Datasets},
  author={Metin Bicer, Andrew TM Phillips, Alessandro Melis, Alison H McGregor, Luca Modenese},
  journal={Journal of Biomechanics},
  volume = {144},
  pages = {111301},
  year = {2022},
  issn = {0021-9290},
  doi={https://doi.org/10.1016/j.jbiomech.2022.111301},
  url = {https://www.sciencedirect.com/science/article/pii/S0021929022003426},
  keywords = {Gait, Deep Learning, Generative Adversarial Networks}
}
```

```bibtex
@article{Bicer2024cGAN,
  title={Generative adversarial networks to create synthetic motion capture datasets including subject and gait characteristics},
  author={Metin Bicer, Andrew TM Phillips, Alessandro Melis, Alison H McGregor, Luca Modenese},
  journal={Journal of Biomechanics},
  volume = {##},
  pages = {112358},
  year = {2024},
  issn = {0021-9290},
  doi={https://doi.org/10.1016/j.jbiomech.2024.112358},
  url = {https://www.sciencedirect.com/science/article/pii/S0021929024004366},
  keywords = {Gait, Synthetic Mocap Dataset, Conditional Generative Adversarial Networks}
}
```

# Our Approach

Here is an example of a real and a synthetic trial from [the first paper](https://doi.org/10.1016/j.jbiomech.2022.111301):

![Animation](Figures/Animation.gif)

Synthetic data were validated by means of statistical comparisons to the experimental data, as presented below from [the second paper](https://doi.org/10.1016/j.jbiomech.2024.112358). This analysis was was carried out using dimensionless walking speed and averaging the sagittal plane kinematics and vertical ground reaction force by dividing trials into walking speed groups.

![Dimensionless](Figures/dimensionless.jpg)


# Requirements
1. create and activate a virtual environment (on Anaconda)  
    using the `environment.yml` file
    ```
	conda env create --file environment.yml
	conda activate cgan
    ```
	**OR** 
	directly from Anaconda by installing the following packages
	```
    conda create --name cgan
    conda activate cgan
    ```
    * torch
    * pandas
    * nump
    * scipy
    * matplotlib
    * ezc3d
    * spm1d
    * sklearn
	* statsmodels
	* spm1d
2. With the environment activated, navigate to the OpenSim installation directory and install OpenSim API
	```
	cd OPENSIM_INSTALL_DIR/sdk/Python
	python setup.py install	
	```

# Run
1. `train_cgans.py`: contains all codes required to train a conditional GAN and generate data with condition(s). Examples are given on lines 675-974. Training data is saved under `data` folder.
2. `compare_train_data.py`: reproduces training dataset characteristics from saved synthetic datasets (under `Results` folder). Condition ranges are given on lines 18-34.
3. `dimensionless_walking_speed_analysis.py`: carries out dimensionless walking speed analysis on the experimental training data and the synthetic dataset generated by the Multi-cGAN. Plots joint angles and vertical ground reaction forces, and compares the two set of data.
4. `dynamical_consistency.py`: reproduces dynamical inconsistency analysis for the WS-cGAN and Multi-cGAN. It can be used for other conditional models if inverse dynamics results are saved under `Results/opensim` folder.
5. `previous_papers.py`: reproduces results from previous papers by the synthetic datasets saved under `Results/papers` folder.
6. `compare_test_data.py`: reproduces plots, error metrics for comparing test experimental data to its synthetic replica generated by all conditional GANs.


